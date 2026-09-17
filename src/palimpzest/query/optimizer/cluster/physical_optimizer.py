from __future__ import annotations

from palimpzest.constants import Cardinality
from palimpzest.core.data.iter_dataset import IterDataset
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.models import (
    GenerationStats,
    OperatorCostEstimates,
    PlanCost,
    SentinelPlanStats,
)
from palimpzest.query.operators.aggregate import AggregateOp
from palimpzest.query.operators.batched import BatchedOperator
from palimpzest.query.operators.convert import ConvertOp, LLMConvert
from palimpzest.query.operators.distinct import DistinctOp
from palimpzest.query.operators.filter import FilterOp, LLMFilter
from palimpzest.query.operators.join import JoinOp as PhysicalJoinOp
from palimpzest.query.operators.logical import BaseScan, ContextScan
from palimpzest.query.operators.logical import JoinOp as LogicalJoinOp
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.operators.scan import ContextScanOp, ScanPhysicalOp
from palimpzest.query.operators.topk import TopKOp
from palimpzest.query.optimizer_config import OptimizerConfig
from palimpzest.query.optimizer.cluster.logical_optimizer import LogicalPlan
from palimpzest.query.optimizer.cluster.physical_operator_clustering import (
    ManualPhysicalOperatorClusteringStrategy,
    PhysicalOperatorCluster,
    PhysicalOperatorClusteringStrategy,
)
from palimpzest.query.optimizer.cluster.physical_operator_selection import (
    PhysicalOperatorSelection,
    PhysicalOperatorSampler,
    ProgressEvent,
)

from palimpzest.query.optimizer.cluster.sampling_budget import (
    EqualClusterSamplingBudgetAllocator,
)
from palimpzest.query.plan import PhysicalPlan, SentinelPlan
from palimpzest.utils.progress import ProgressManager, create_progress_manager
from palimpzest.validator.validator import Validator


class PhysicalOptimizer:
    """Physical optimizer for clustered candidate search.

    This optimizer currently builds the physical candidate clusters and the
    stateful selector that will sample them. Candidate instantiation, clustering,
    and selection are kept as separate pieces so each can evolve independently.
    """

    def __init__(
        self,
        optimizer_config: OptimizerConfig,
        clustering_strategy: PhysicalOperatorClusteringStrategy | None = None,
        max_workers: int = 64,
        progress: bool = True,
    ):
        """Initialize optimizer state from runtime optimizer configuration."""
        self.optimizer_config = optimizer_config
        self.policy = optimizer_config.policy
        self.max_workers = max(max_workers, 1)
        self.progress = progress
        self.progress = False

        if optimizer_config.budget_allocator is None:
            self.budget_allocator = EqualClusterSamplingBudgetAllocator(
                optimizer_config.sample_budget
            )
        else:
            self.budget_allocator = optimizer_config.budget_allocator

        self.clustering_strategy = (
            ManualPhysicalOperatorClusteringStrategy()
            if clustering_strategy is None
            else clustering_strategy
        )

    def build_sampling_progress_plan(
        self,
        logical_plan: LogicalPlan,
        op_clusters: dict[str, PhysicalOperatorCluster],
    ) -> SentinelPlan:
        """Build a display-only sentinel plan for cluster sampling progress."""
        source_op_ids = logical_plan.source_op_ids
        progress_plans = {}
        for logical_op_id in logical_plan.topological_order:
            subplans = [
                progress_plans[source_op_id]
                for source_op_id in source_op_ids[logical_op_id]
            ]
            progress_plans[logical_op_id] = SentinelPlan(
                op_clusters[logical_op_id].physical_ops,
                subplans=subplans,
            )

        return progress_plans[logical_plan.root_op_id]

    def update_sampling_progress_total(
        self,
        progress_manager: ProgressManager,
        progress_logical_op_id: str,
        samples_drawn: int,
        progress_total: int,
        op_sample_budget: int,
    ) -> None:
        """Shrink progress totals when an operator exhausts samples early."""
        if not hasattr(progress_manager, "op_progress"):
            return

        task = progress_manager.unique_logical_op_id_to_task.get(progress_logical_op_id)
        if task is not None:
            progress_manager.op_progress.update(task, total=samples_drawn)

        adjusted_total = progress_total - (op_sample_budget - samples_drawn)
        progress_manager.overall_progress.update(
            progress_manager.overall_task_id,
            total=max(adjusted_total, 0),
            refresh=True,
        )
        progress_manager.live_display.refresh()

    def build_physical_plan(
        self,
        logical_plan: LogicalPlan,
        selected_physical_ops: dict[str, PhysicalOperator],
        selected_op_cost_estimates: dict[str, OperatorCostEstimates],
    ) -> PhysicalPlan:
        """Assemble a ``PhysicalPlan`` from the selected physical operators."""
        source_op_ids = logical_plan.source_op_ids
        physical_plans = {}
        for logical_op_id in logical_plan.topological_order:
            source_ids = source_op_ids[logical_op_id]
            subplans = [physical_plans[source_op_id] for source_op_id in source_ids]
            physical_op = selected_physical_ops[logical_op_id]
            op_cost_estimates = selected_op_cost_estimates[logical_op_id]

            if len(source_ids) == 0:
                source_cost_estimates = selected_op_cost_estimates[logical_op_id]
                right_source_cost_estimates = None
            elif len(source_ids) == 1:
                source_cost_estimates = selected_op_cost_estimates[source_ids[0]]
                right_source_cost_estimates = None
            elif len(source_ids) == 2:
                source_cost_estimates = selected_op_cost_estimates[source_ids[0]]
                right_source_cost_estimates = selected_op_cost_estimates[source_ids[1]]
            else:
                raise ValueError(
                    f"Logical op {logical_op_id} has {len(subplans)} source plans"
                )

            input_cardinality = source_cost_estimates.cardinality
            if right_source_cost_estimates is not None:
                input_cardinality *= right_source_cost_estimates.cardinality
            op_plan_cost = PlanCost(
                cost=op_cost_estimates.cost_per_record * input_cardinality,
                time=op_cost_estimates.time_per_record * input_cardinality,
                quality=op_cost_estimates.quality,
                op_estimates=op_cost_estimates,
            )

            if len(subplans) == 0:
                plan_cost = op_plan_cost
            elif len(subplans) == 1:
                plan_cost = subplans[0].plan_cost + op_plan_cost
            elif len(subplans) == 2:
                optimizer_execution_strategy = self.optimizer_config.execution_strategy
                is_parallel_execution = (
                    optimizer_execution_strategy.is_fully_parallel()
                    if hasattr(optimizer_execution_strategy, "is_fully_parallel")
                    else str(optimizer_execution_strategy).lower() == "parallel"
                )
                execution_strategy = (
                    "parallel" if is_parallel_execution else "sequential"
                )
                plan_cost = op_plan_cost.join_add(
                    subplans[0].plan_cost,
                    subplans[1].plan_cost,
                    execution_strategy=execution_strategy,
                )

            physical_plans[logical_op_id] = PhysicalPlan(
                physical_op,
                subplans=subplans,
                plan_cost=plan_cost,
            )

        return physical_plans[logical_plan.root_op_id]

    def optimize(
        self,
        logical_plan: LogicalPlan,
        optimization_stats: SentinelPlanStats,
        validator: Validator | None = None,
    ) -> PhysicalPlan:
        """Optimize a logical plan and return a physical plan.

        First, for each operator in the logical plan, we generate a clustering of all its physical operator candidates, and the overall sampling budget is allocated to each logical operator in the plan.
        Then, each physical operator cluster is sampled according to the allocated budget, and the observed execution statistics are used to select the best physical operator for each logical operator.
        Finally, a physical plan is constructed from the selected physical operators and their cost estimates.

        """

        source_op_ids = logical_plan.source_op_ids
        topological_order = logical_plan.topological_order

        # op_clusters is a dict[str, PhysicalOperatorCluster], keyed by logical op id
        op_clusters = self.clustering_strategy.initialize_clusters(
            logical_plan, self.optimizer_config, optimization_stats
        )

        sample_budgets = self.budget_allocator(logical_plan)

        # Logic to track progress
        progress_manager = None
        progress_logical_op_ids = {}

        update_progress = lambda event: None  # Default no-op if progress is disabled
        if self.progress:
            progress_plan = self.build_sampling_progress_plan(
                logical_plan,
                op_clusters,
            )
            progress_manager = create_progress_manager(
                progress_plan,
                sample_budget=sum(sample_budgets.values()),
                sample_cost_budget=None,
                progress=self.progress,
            )
            for idx, (progress_logical_op_id, _) in enumerate(progress_plan):
                logical_op_id = topological_order[idx]
                unique_progress_logical_op_id = f"{idx}-{progress_logical_op_id}"
                progress_logical_op_ids[logical_op_id] = unique_progress_logical_op_id
                task = progress_manager.unique_logical_op_id_to_task[
                    unique_progress_logical_op_id
                ]
                if task is not None:
                    progress_manager.op_progress.update(
                        task,
                        total=sample_budgets[logical_op_id],
                    )

            progress_manager.start()
            def update_progress(event: ProgressEvent) -> None:
                progress_manager.incr(
                    progress_logical_op_ids[event.logical_op_id],
                    event.samples_drawn,
                    display_text=
                        f"Sampling {event.samples_drawn}/{event.total_samples} for logical op {event.logical_op_id}"
                    total_cost=event.incremental_cost
                )
        try:
            sampler = PhysicalOperatorSampler(
                policy=self.policy,
                total_sampling_rounds=self.optimizer_config.sample_budget,
                seed=self.optimizer_config.seed,
                max_workers=self.max_workers
            )

            sampled_records_by_logical_op_id: dict[str, list[DataRecord]] = {}
            progress_total = sum(sample_budgets.values())
            source_op_ids = logical_plan.source_op_ids

            results = {} # intermediate storage for data records results per op
            estimates = {} # intermediate storage for cost estimates per op
            for logical_op_id in logical_plan.topological_order:
                logical_op = logical_plan.operators[logical_op_id]
                cluster = op_clusters[logical_op_id]
                source_id = source_op_ids[logical_op_id]
                

                input_records = results.get(source_id, [])

                # If this is a source operator, the cardinality is found 
                # in the cost estimates of the first cluster path node. 
                # Otherwise, the source cost estimates are found in the cost estimates of the source operator clusters.
                if len(source_id) == 0:
                    input_estimates = [
                        selection.cluster_path[0].cost_estimates
                    ]
                else:
                    input_estimates = [estimates[source_id]]

                records, estimates = sampler.sample_physical_operator_cluster(
                    cluster,
                    sample_budget=sample_budgets[logical_op_id],
                    input_records=input_records,
                    input_estimates=input_estimates,
                    optimization_stats=optimization_stats,
                    validator=validator,
                    on_sample_completed=update_progress
                )
                results[logical_op_id] = records
                estimates[logical_op_id] = estimates
        finally:
            if progress_manager is not None:
                progress_manager.finish()

        selected_physical_ops, selected_op_cost_estimates = (
            self.select_physical_operators(
                logical_plan,
                op_clusters,
                sampler,
            )
        )

        optimized_plan = self.build_physical_plan(
            logical_plan,
            selected_physical_ops,
            selected_op_cost_estimates,
        )

        return optimized_plan

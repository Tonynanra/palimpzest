"""Utilities for grouping flat physical operator candidates into search clusters."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional, TypeAlias

from palimpzest.constants import NAIVE_BYTES_PER_RECORD
from palimpzest.core.models import (
    OperatorCostEstimates,
    SentinelPlanStats,
    OperatorStats,
)
from palimpzest.query.operators.aggregate import SemanticAggregate
from palimpzest.query.operators.batched import BatchedFilter
from palimpzest.query.operators.convert import LLMConvertBonded
from palimpzest.query.operators.critique_and_refine import (
    CritiqueAndRefineConvert,
    CritiqueAndRefineFilter,
)
from palimpzest.query.operators.filter import LLMFilter
from palimpzest.query.operators.image_filter import RescaledImageFilter
from palimpzest.query.operators.join import EmbeddingJoin, JoinOp, NestedLoopsJoin
from palimpzest.query.operators.logical import JoinOp as LogicalJoinOp
from palimpzest.query.operators.logical import BaseScan, ContextScan, LogicalOperator
from palimpzest.query.operators.mixture_of_agents import (
    MixtureOfAgentsConvert,
    MixtureOfAgentsFilter,
)
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.operators.rag import RAGConvert, RAGFilter
from palimpzest.query.operators.scan import MarshalAndScanDataOp
from palimpzest.query.operators.split import SplitConvert, SplitFilter
from palimpzest.query.operators.topk import TopKOp
from palimpzest.query.optimizer.cluster.logical_optimizer import LogicalPlan
from palimpzest.query.optimizer.cluster.physical_operator_registry import (
    find_physical_candidates,
)
from palimpzest.query.optimizer_config import OptimizerConfig

# One grouping layer in a manual cluster spec. The first tuple entry is the
# display name; the second is either a physical operator attribute name or a
# callable that extracts the value to group by.
ClusterDimension: TypeAlias = tuple[str, str | Callable[[PhysicalOperator], object]]

# Physical operator class to ordered grouping dimensions.
PhysicalOperatorClusterSpec: TypeAlias = dict[
    type[PhysicalOperator],
    list[ClusterDimension],
]


@dataclass
class PhysicalOperatorCluster:
    """A set of physical operator candidates, potentially stored within children clusters.
    Leaf clusters only contain a single physical operator.
    Parent clusters contain one or more children clusters, each of which may contain multiple physical operators.

    Every node represents the concrete physical operators in ``physical_ops``.
    Clusters keep a ``cost_estimates`` object.
    Parent clusters compute it averaging through the estimates of their children, while leaf clusters keep the estimates of their physical operators.
    """

    name: str
    logical_op: LogicalOperator
    physical_ops: list[PhysicalOperator]
    cost_estimates: OperatorCostEstimates
    children: list[PhysicalOperatorCluster] = field(default_factory=list)

    def __str__(self) -> str:
        """Render the cluster tree without listing concrete physical operators."""
        lines = []
        stack = [(self, 0)]
        while len(stack) > 0:
            cluster, depth = stack.pop()
            indent = "  " * depth
            cost_suffix = ""
            if cluster.cost_estimates is not None:
                cost_estimates = cluster.cost_estimates
                cost_suffix = (
                    " "
                    f"(card={cost_estimates.cardinality:.4g}, "
                    f"time/rec={cost_estimates.time_per_record:.4g}, "
                    f"cost/rec={cost_estimates.cost_per_record:.4g}, "
                    f"quality={cost_estimates.quality:.4g})"
                )

            if len(cluster.children) == 0:
                lines.append(f"{indent}{cluster.name}{cost_suffix}")
                lines.append(
                    f"{indent}  -> ... ({len(cluster.physical_ops)} physical operators)"
                )
            else:
                lines.append(f"{indent}{cluster.name}{cost_suffix}")
                stack.extend((child, depth + 1) for child in reversed(cluster.children))

        return "\n".join(lines)


MANUAL_PHYSICAL_OPERATOR_CLUSTER_SPEC: PhysicalOperatorClusterSpec = {
    # First layer is always the physical operator class. These dimensions define
    # the hand-written layers below each class.
    LLMFilter: [("model", "model")],
    BatchedFilter: [("batch_size", "batch_size"), ("model", "model")],
    RescaledImageFilter: [("rescale_factor", "rescale_factor"), ("model", "model")],
    RAGFilter: [
        ("chunk_size", "chunk_size"),
        ("num_chunks_per_field", "num_chunks_per_field"),
        ("model", "model"),
        ("embedding_model", "embedding_model"),
    ],
    SplitFilter: [
        ("num_chunks", "num_chunks"),
        ("min_size_to_chunk", "min_size_to_chunk"),
        ("model", "model"),
    ],
    MixtureOfAgentsFilter: [
        ("num_proposer_models", lambda op: len(op.proposer_models)),
        ("aggregator_model", "aggregator_model"),
        ("temperatures", lambda op: tuple(op.temperatures)),
        ("proposer_models", lambda op: tuple(op.proposer_models)),
    ],
    CritiqueAndRefineFilter: [
        ("model", "model"),
        ("critic_model", "critic_model"),
        ("refine_model", "refine_model"),
    ],
    LLMConvertBonded: [("model", "model")],
    RAGConvert: [
        ("chunk_size", "chunk_size"),
        ("num_chunks_per_field", "num_chunks_per_field"),
        ("model", "model"),
        ("embedding_model", "embedding_model"),
    ],
    SplitConvert: [
        ("num_chunks", "num_chunks"),
        ("min_size_to_chunk", "min_size_to_chunk"),
        ("model", "model"),
    ],
    MixtureOfAgentsConvert: [
        ("num_proposer_models", lambda op: len(op.proposer_models)),
        ("aggregator_model", "aggregator_model"),
        ("temperatures", lambda op: tuple(op.temperatures)),
        ("proposer_models", lambda op: tuple(op.proposer_models)),
    ],
    CritiqueAndRefineConvert: [
        ("model", "model"),
        ("critic_model", "critic_model"),
        ("refine_model", "refine_model"),
    ],
    NestedLoopsJoin: [("model", "model"), ("join_parallelism", "join_parallelism")],
    EmbeddingJoin: [
        ("num_samples", "num_samples"),
        ("embedding_model", "embedding_model"),
        ("model", "model"),
        ("join_parallelism", "join_parallelism"),
    ],
    SemanticAggregate: [("model", "model")],
    TopKOp: [("k", "k")],
}


class PhysicalOperatorClusteringStrategy:
    """Interface for alternative physical operator clustering methods."""

    def initialize_clusters(
        self,
        logical_plan: LogicalPlan,
        optimizer_config: OptimizerConfig,
        optimization_stats: SentinelPlanStats,
    ) -> dict[str, PhysicalOperatorCluster]:
        """This method is given a logical plan, which is a DAG of logical operators,
        and creates, for each of the logical operators, clusters of
        physical operators that can implement the logical operator.
        The method initializes the cost estimates for each physical operator within the clusters
        based on the cardinalities and cost estimates of its source operators.
        To do this, it has to know the full plan topology.

        The method topologically walks the logical plan from source scans
        to downstream consumers, passing each logical operator's source cluster
        cost estimates into the clustering strategy. Base scans seed their
        source estimate from ``len(datasource)``; context scans seed cardinality
        ``1.0``; joins receive left and right source estimates.
        """

        # sources is keyed by logical operator ID and keeps the op id of the sources
        source_ids = logical_plan.source_op_ids  # dict[str, list[str]]
        op_clusters = {}
        cost_estimates = {}
        optimization_stats.operator_stats = {}

        for op_id in logical_plan.topological_order:
            logical_op = logical_plan.operators[op_id]
            source_id = source_ids[op_id]
            optimization_stats.operator_stats[op_id] = {}

            if isinstance(logical_op, BaseScan):
                source_estimates = [
                    OperatorCostEstimates(
                        cardinality=len(logical_op.datasource),  # type: ignore
                        time_per_record=0.0,
                        cost_per_record=0.0,
                        quality=1.0,
                    )
                ]
            elif isinstance(logical_op, ContextScan):
                source_estimates = [
                    OperatorCostEstimates(
                        cardinality=1.0,
                        time_per_record=0.0,
                        cost_per_record=0.0,
                        quality=1.0,
                    )
                ]
            elif len(source_id) == 1:
                source_estimates = [cost_estimates[source_id[0]]]
            elif isinstance(logical_op, LogicalJoinOp):
                if len(source_id) != 2:
                    raise ValueError(f"Join op {op_id} has {len(source_id)} sources")
                left, right = source_id
                source_estimates = [cost_estimates[left], cost_estimates[right]]
            else:
                raise ValueError(f"Op {op_id} has {len(source_id)} source operators")

            cluster = self.build_cluster(logical_op, optimizer_config, source_estimates)
            op_clusters[op_id] = cluster
            cost_estimates[op_id] = cluster.cost_estimates

            for physical_op in op_clusters[op_id].physical_ops:
                physical_op_id = physical_op.get_full_op_id()
                optimization_stats.operator_stats[op_id][physical_op_id] = (
                    OperatorStats(
                        full_op_id=op_id,
                        op_name=physical_op.op_name(),
                        source_unique_logical_op_ids=source_ids[op_id],
                        plan_id=optimization_stats.plan_id,
                        op_details={
                            key: str(value)
                            for key, value in physical_op.get_id_params().items()
                        },
                    )
                )

        return op_clusters

    def build_cluster(
        self,
        logical_op: LogicalOperator,
        optimizer_config: OptimizerConfig,
        source_op_cost_estimates: list[OperatorCostEstimates] | None = None,
    ) -> PhysicalOperatorCluster:
        """Build a cluster tree for one logical operator's physical candidates."""
        raise NotImplementedError


class ManualPhysicalOperatorClusteringStrategy(PhysicalOperatorClusteringStrategy):
    """Build a cluster tree from the hand-written physical operator spec.

    The registry owns candidate instantiation; this strategy only organizes an
    already-instantiated flat list. It computes naive costs for each concrete
    operator, stores those estimates at leaf nodes, and stores averaged estimates
    at every internal cluster.
    """

    def __init__(
        self,
        cluster_spec: PhysicalOperatorClusterSpec | None = None,
        input_record_size_in_bytes: int | float = NAIVE_BYTES_PER_RECORD,
    ):
        """Create a manual clustering strategy.

        ``cluster_spec`` controls the ordered dimensions below each physical
        operator class. ``input_record_size_in_bytes`` is only used by scan
        operators whose naive estimates require an input record size.
        """
        self.cluster_spec = (
            MANUAL_PHYSICAL_OPERATOR_CLUSTER_SPEC
            if cluster_spec is None
            else cluster_spec
        )
        self.input_record_size_in_bytes = input_record_size_in_bytes

    def build_cluster(
        self,
        logical_op: LogicalOperator,
        optimizer_config: OptimizerConfig,
        source_op_cost_estimates: list[OperatorCostEstimates] | None = None,
    ) -> PhysicalOperatorCluster:
        """Cluster concrete physical operators for a single logical operator.

        The root node is named after the logical operator. Its first layer is
        the physical operator class, followed by dimensions from the manual
        spec. Each concrete physical operator is stored as a leaf cluster with
        its own cost estimate, and every parent receives averaged child estimates.
        """

        physical_ops = find_physical_candidates(logical_op, optimizer_config)
        if source_op_cost_estimates is None:
            source_op_cost_estimates = [
                OperatorCostEstimates(
                    cardinality=100,
                    time_per_record=0.0,
                    cost_per_record=0.0,
                    quality=1.0,
                )
            ]

        op_to_cost_estimates = {}
        for op in physical_ops:
            if isinstance(op, MarshalAndScanDataOp):
                op_to_cost_estimates[op.get_full_op_id()] = op.naive_cost_estimates(
                    source_op_cost_estimates[0],
                    input_record_size_in_bytes=self.input_record_size_in_bytes,
                )
            elif isinstance(op, JoinOp):
                assert len(source_op_cost_estimates) == 2
                left_cost, right_cost = source_op_cost_estimates
                op_to_cost_estimates[op.get_full_op_id()] = op.naive_cost_estimates(
                    left_cost, right_cost
                )
            else:
                op_to_cost_estimates[op.get_full_op_id()] = op.naive_cost_estimates(
                    source_op_cost_estimates[0]
                )

        class_to_ops = defaultdict(list)
        for op in physical_ops:
            class_to_ops[type(op)].append(op)

        children = []
        for op_class, ops in sorted(
            class_to_ops.items(), key=lambda item: item[0].__name__
        ):
            dimensions = self.cluster_spec.get(op_class, [])
            dimension_children = self._build_dimension_clusters(
                logical_op,
                ops,
                dimensions,
                op_to_cost_estimates,
            )
            cluster_cost_estimates = self._average_cost_estimates(
                [
                    child.cost_estimates
                    for child in dimension_children
                    if child.cost_estimates is not None
                ]
            )
            children.append(
                PhysicalOperatorCluster(
                    name=op_class.__name__,
                    logical_op=logical_op,
                    physical_ops=ops,
                    children=dimension_children,
                    cost_estimates=cluster_cost_estimates,
                )
            )

        return PhysicalOperatorCluster(
            name=logical_op.logical_op_name(),
            logical_op=logical_op,
            physical_ops=physical_ops,
            children=children,
            cost_estimates=self._average_cost_estimates(
                [
                    child.cost_estimates
                    for child in children
                    if child.cost_estimates is not None
                ]
            ),
        )

    def _build_dimension_clusters(
        self,
        logical_op: LogicalOperator,
        physical_ops: list[PhysicalOperator],
        dimensions: list[ClusterDimension],
        op_to_cost_estimates: dict[str, OperatorCostEstimates],
    ) -> list[PhysicalOperatorCluster]:
        """Recursively group operators by the remaining manual dimensions."""
        if len(dimensions) == 0:
            return [
                PhysicalOperatorCluster(
                    name=op.get_full_op_id(),
                    logical_op=logical_op,
                    physical_ops=[op],
                    cost_estimates=op_to_cost_estimates[op.get_full_op_id()],
                )
                for op in sorted(physical_ops, key=lambda op: op.get_full_op_id())
            ]

        dimension_name, accessor = dimensions[0]
        value_to_ops = defaultdict(list)
        for op in physical_ops:
            value = getattr(op, accessor) if isinstance(accessor, str) else accessor(op)
            if isinstance(value, list):
                value = tuple(value)
            value_to_ops[value].append(op)

        children = []
        for value, ops in sorted(value_to_ops.items(), key=lambda item: str(item[0])):
            next_children = self._build_dimension_clusters(
                logical_op,
                ops,
                dimensions[1:],
                op_to_cost_estimates,
            )
            cluster_cost_estimates = self._average_cost_estimates(
                [
                    child.cost_estimates
                    for child in next_children
                    if child.cost_estimates is not None
                ]
            )
            children.append(
                PhysicalOperatorCluster(
                    name=f"{dimension_name}={value}",
                    logical_op=logical_op,
                    physical_ops=ops,
                    children=next_children,
                    cost_estimates=cluster_cost_estimates,
                )
            )

        return children

    def _average_cost_estimates(
        self,
        cost_estimates: list[OperatorCostEstimates],
    ) -> OperatorCostEstimates:
        """Average every populated ``OperatorCostEstimates`` field independently."""

        averaged_fields = {}
        for field_name in OperatorCostEstimates.model_fields:
            values = [
                getattr(cost_estimate, field_name)
                for cost_estimate in cost_estimates
                if getattr(cost_estimate, field_name) is not None
            ]
            averaged_fields[field_name] = (
                sum(values) / len(values) if len(values) > 0 else None
            )

        return OperatorCostEstimates(**averaged_fields)

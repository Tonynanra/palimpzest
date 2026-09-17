"""Selection policy for sampling physical operators from cluster trees."""

from __future__ import annotations
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Iterable

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import List

from palimpzest.constants import Cardinality
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.models import (
    GenerationStats,
    OperatorCostEstimates,
    PlanCost,
    SentinelPlanStats,
)
from palimpzest.policy import MaxQuality, MinCost, MinTime, Policy
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
from palimpzest.query.optimizer.cluster.physical_operator_clustering import (
    PhysicalOperatorCluster,
)
from palimpzest.query.optimizer.cluster.logical_optimizer import LogicalPlan
from palimpzest.validator.validator import Validator


@dataclass
class PhysicalOperatorSelection:
    """A sampled physical operator and the cluster branch that produced it.

    The cluster_path is a list of clusters from the root down to the leaf
    that contains the selected physical operator. The leaf cluster is the last
    element; internal clusters are its ancestors. The path is used during
    record_execution() to update sample counts and propagate new cost
    estimates up the tree.
    """

    record_id: str
    physical_op: PhysicalOperator
    cluster_path: list[PhysicalOperatorCluster]


@dataclass
class ProgressEvent:
    """Event emitted after each physical operator sample is executed.

    This event can be used to update progress bars or logs in the caller.
    """

    logical_op_id: str
    output_count: int
    incremental_cost: float


class PhysicalOperatorSampler:
    """Stateful sampler for the clustered physical operator search space.

    The sampler owns state that changes across sampling rounds: how many times
    each cluster branch has been sampled, and which physical operators have
    already been executed for each input record. It chooses a branch using the
    exploration/exploitation score, then records executions back into the
    affected cluster path.

    The physical operator sampler keeps a running state of the physical ops
    that have been sampled and their observed estimates/qualities.

    Key mutable state:
      - cluster_sample_counts: total number of times each cluster (identified by id) has been visited during sampling.
      - cluster_input_record_counts: number of input records considered for each cluster, used as a normalizer for confidence.
      - physical_op_sample_counts: how many times each concrete physical operator (by full op id) has been sampled.
      - executed_physical_op_ids_by_record: per-record set of physical operator ids already executed (to avoid re-selection).
      - physical_op_quality_samples: list of observed quality values per physical operator (maintained during record_execution).

    The sampler uses a tree of PhysicalOperatorCluster nodes. Every cluster's
    physical_ops lists the concrete operators under that subtree. Leaf clusters
    store the estimate for their concrete physical operator in cost_estimates;
    internal clusters aggregate child cost_estimates. The cluster_path in a
    PhysicalOperatorSelection tracks the branch from root to leaf that produced
    a chosen operator.

    After all sampling rounds, select_best_physical_operator() performs a final
    exploitation step by evaluating all leaf operators with uncertainty_aware_select.
    """

    def __init__(
        self,
        policy: Policy,
        total_sampling_rounds: int,
        max_input_records: int = 1,
        seed: int = 42,
        max_workers: int = 8,
    ):
        """Create a selector for a fixed policy and sampling budget.

        max_input_records is used as the default confidence normalizer when
        a caller selects a physical operator for an already-chosen record.
        """
        self.policy = policy
        self.total_sampling_rounds = total_sampling_rounds
        self.max_input_records = max_input_records
        self.rng = random.Random(seed)
        self.cluster_sample_counts = defaultdict(int)
        self.cluster_input_record_counts = defaultdict(lambda: self.max_input_records)
        self.physical_op_sample_counts = defaultdict(int)
        self.executed_physical_op_ids_by_record = defaultdict(set)

        self.physical_op_quality_samples = defaultdict(list)
        self.max_workers = max_workers

    def uncertainty_aware_select(
        self,
        leaf_cost_estimates_by_physical_op_id: dict[str, OperatorCostEstimates],
        physical_op_confidences: dict[str, float],
        uncertainty_weight: float = 0.0,
    ) -> tuple[str, OperatorCostEstimates]:
        """Return the physical operator id with the best final score.
        The score for each operator is
            policy_score - uncertainty_weight * (1 - confidence)
        Policy scores are normalized to [0,1] based on the best and worst metric
        among the candidates. In case of a tie, the policy's choose() method
        breaks it.

        This strategy is used by PhysicalOperatorSelector.select_best_physical_operator()
        after all sampling rounds are complete. It combines the raw policy score
        (based on quality, cost, or time) with an uncertainty penalty derived from
        the confidence (fraction of samples collected) for each operator. The final
        selection picks the operator with the highest composite score.

        Supported policies: MaxQuality, MinCost, MinTime.
        """

        if len(leaf_cost_estimates_by_physical_op_id) == 0:
            raise ValueError("Cannot select from an empty physical operator set")
        weight = max(uncertainty_weight, 0.0)

        metric_values = {}
        for (
            physical_op_id,
            cost_estimates,
        ) in leaf_cost_estimates_by_physical_op_id.items():
            if isinstance(self.policy, MaxQuality):
                metric_values[physical_op_id] = cost_estimates.quality
            elif isinstance(self.policy, MinCost):
                metric_values[physical_op_id] = cost_estimates.cost_per_record
            elif isinstance(self.policy, MinTime):
                metric_values[physical_op_id] = cost_estimates.time_per_record

        min_value = min(metric_values.values())
        max_value = max(metric_values.values())

        best_physical_op_id = None
        best_cost_estimates = None
        best_score = None
        for (
            physical_op_id,
            cost_estimates,
        ) in leaf_cost_estimates_by_physical_op_id.items():
            value = metric_values[physical_op_id]
            if min_value == max_value:
                policy_score = 1.0
            elif isinstance(self.policy, MaxQuality):
                policy_score = (value - min_value) / (max_value - min_value)
            else:
                policy_score = (max_value - value) / (max_value - min_value)

            confidence = physical_op_confidences.get(physical_op_id, 0.0)
            uncertainty = 1.0 - max(0.0, min(confidence, 1.0))
            score = policy_score - (weight * uncertainty)
            if best_score is None or score > best_score:
                best_physical_op_id = physical_op_id
                best_cost_estimates = cost_estimates
                best_score = score
            elif score == best_score:
                plan_cost = PlanCost(
                    cost=cost_estimates.cost_per_record,
                    time=cost_estimates.time_per_record,
                    quality=cost_estimates.quality,
                )
                assert best_cost_estimates is not None
                best_plan_cost = PlanCost(
                    cost=best_cost_estimates.cost_per_record,
                    time=best_cost_estimates.time_per_record,
                    quality=best_cost_estimates.quality,
                )
                if self.policy.choose(plan_cost, best_plan_cost):
                    best_physical_op_id = physical_op_id
                    best_cost_estimates = cost_estimates

        assert best_physical_op_id is not None and best_cost_estimates is not None
        return best_physical_op_id, best_cost_estimates

    def select_next_sample(
        self,
        root_cluster: PhysicalOperatorCluster,
        input_record_ids: list[str],
        sampling_round_idx: int,
    ) -> PhysicalOperatorSelection:
        """Choose both a random input record and a physical operator for it.

        Records for which every physical operator has already been executed are
        excluded before sampling the record.  If the root cluster contains
        exactly one physical operator, that operator is returned directly with a
        cluster_path that walks down to its leaf (following children that contain
        the same operator id).
        """
        available_record_ids = [
            record_id
            for record_id in input_record_ids
            if self._cluster_has_available_physical_op(root_cluster, record_id)
        ]
        if len(available_record_ids) == 0:
            raise ValueError(
                f"All physical operators in cluster {root_cluster.name} have already "
                "been executed on every candidate input record"
            )
        record_id = self.rng.choice(input_record_ids)

        # If the root cluster only has one physical operator, skip selection but
        # preserve the path to the leaf that owns the operator cost estimates.
        if len(root_cluster.physical_ops) == 1:
            physical_op = root_cluster.physical_ops[0]
            physical_op_id = physical_op.get_full_op_id()
            cluster = root_cluster
            cluster_path = [root_cluster]
            while len(cluster.children) > 0:
                matching_children = [
                    child
                    for child in cluster.children
                    if any(
                        op.get_full_op_id() == physical_op_id
                        for op in child.physical_ops
                    )
                ]
                cluster = matching_children[0]
                cluster_path.append(cluster)

            return PhysicalOperatorSelection(
                record_id=record_id,
                physical_op=physical_op,
                cluster_path=cluster_path,
            )

        return self.select_physical_operator(
            root_cluster,
            record_id,
            sampling_round_idx,
            max_input_records=len(input_record_ids),
        )

    def select_physical_operator(
        self,
        root_cluster: PhysicalOperatorCluster,
        record_id: str | int,
        sampling_round_idx: int,
        max_input_records: int | None = None,
    ) -> PhysicalOperatorSelection:
        """Choose a physical operator by walking from root cluster to leaf.

        At each internal node, child weights combine uncertainty and policy
        score: alpha * (1 - confidence) + (1 - alpha) * predicted_score.
        alpha decreases as the sampling budget is consumed.

        The walking stops when a leaf cluster is reached (no children). Among
        the leaf's available physical operators (those not yet executed for this
        record), one is chosen uniformly at random.
        """
        if not self._cluster_has_available_physical_op(root_cluster, record_id):
            raise ValueError(
                f"All physical operators in cluster {root_cluster.name} have already "
                f"been executed on record {record_id}"
            )

        alpha = (self.total_sampling_rounds - sampling_round_idx - 1) / max(
            self.total_sampling_rounds,
            1,
        )
        alpha = max(0.0, min(1.0, alpha))
        if max_input_records is not None:
            self.cluster_input_record_counts[id(root_cluster)] = max_input_records

        cluster = root_cluster
        cluster_path = [root_cluster]
        while len(cluster.children) > 0:
            available_children = [
                child
                for child in cluster.children
                if self._cluster_has_available_physical_op(child, record_id)
            ]
            policy_scores = self._policy_scores(available_children)
            weights = []
            for child in available_children:
                confidence = self.cluster_confidence(
                    child,
                    max_input_records=max_input_records,
                )
                predicted_score = policy_scores[id(child)]
                weights.append(
                    alpha * (1.0 - confidence) + (1.0 - alpha) * predicted_score
                )

            cluster = self._weighted_choice(available_children, weights)
            cluster_path.append(cluster)

        available_physical_ops = [
            op
            for op in cluster.physical_ops
            if op.get_full_op_id()
            not in self.executed_physical_op_ids_by_record[record_id]
        ]
        physical_op = self.rng.choice(available_physical_ops)
        return PhysicalOperatorSelection(
            record_id=record_id,
            physical_op=physical_op,
            cluster_path=cluster_path,
        )

    def record_execution(
        self,
        selection: PhysicalOperatorSelection,
        cost_estimates: OperatorCostEstimates | None = None,
    ) -> None:
        """Record an execution and refresh scores for the affected branch.

        The selected physical operator is blacklisted for the selected record,
        every cluster on the selected path gets one additional sample, and the
        path's cost estimates are recomputed from leaf to root.

        If cost_estimates is None, the leaf cluster's stored estimate is used.
        The observed quality value is appended to
        physical_op_quality_samples for later entropy-based confidence
        calculations.
        """
        physical_op_id = selection.physical_op.get_full_op_id()
        self.executed_physical_op_ids_by_record[selection.record_id].add(physical_op_id)
        self.physical_op_sample_counts[physical_op_id] += 1

        leaf_cluster = selection.cluster_path[-1]
        if cost_estimates is None:
            cost_estimates = leaf_cluster.cost_estimates
        if cost_estimates is None:
            raise ValueError(
                f"Leaf cluster for physical op {physical_op_id} has no cost estimates"
            )
        leaf_cluster.cost_estimates = cost_estimates

        for cluster in selection.cluster_path:
            self.cluster_sample_counts[id(cluster)] += 1

        # Propagate new aggregated cost estimates up the tree.
        for cluster in reversed(selection.cluster_path):
            if len(cluster.children) > 0:
                cluster.cost_estimates = self._average_cost_estimates(
                    [
                        child.cost_estimates
                        for child in cluster.children
                        if child.cost_estimates is not None
                    ]
                )

        self.physical_op_quality_samples[physical_op_id].append(cost_estimates.quality)

    def select_best_physical_operator(
        self,
        root_cluster: PhysicalOperatorCluster,
    ) -> tuple[PhysicalOperator, OperatorCostEstimates]:
        """Return the best concrete physical operator under a cluster.

        This is the final exploitation step after sampling has updated the
        cluster tree.  It collects all leaf operators' cost estimates,
        computes confidences as the fraction of maximum possible samples
        collected per operator, and delegates to uncertainty_aware_select() to pick the best operator.

        Only single-objective policies (MaxQuality, MinCost, MinTime)
        are currently supported.
        """
        if not isinstance(self.policy, (MaxQuality, MinCost, MinTime)):
            raise NotImplementedError(
                f"Unsupported physical operator selection policy: {type(self.policy).__name__}"
            )

        physical_op_by_id = {
            op.get_full_op_id(): op for op in root_cluster.physical_ops
        }
        leaf_cost_estimates_by_physical_op_id = {}
        stack = [root_cluster]
        while len(stack) > 0:
            cluster = stack.pop()
            if len(cluster.children) == 0:
                if cluster.cost_estimates is not None:
                    for op in cluster.physical_ops:
                        leaf_cost_estimates_by_physical_op_id[op.get_full_op_id()] = (
                            cluster.cost_estimates
                        )
            else:
                stack.extend(cluster.children)

        if len(leaf_cost_estimates_by_physical_op_id) == 0:
            raise ValueError(f"Cluster {root_cluster.name} has no physical operators")

        max_input_records = self.cluster_input_record_counts[id(root_cluster)]
        physical_op_confidences = {
            physical_op_id: min(
                self.physical_op_sample_counts[physical_op_id]
                / max(max_input_records, 1),
                1.0,
            )
            for physical_op_id in leaf_cost_estimates_by_physical_op_id
        }
        best_physical_op_id, best_cost_estimates = self.uncertainty_aware_select(
            leaf_cost_estimates_by_physical_op_id,
            physical_op_confidences,
        )

        return physical_op_by_id[best_physical_op_id], best_cost_estimates

    def cluster_confidence(
        self,
        cluster: PhysicalOperatorCluster,
        max_input_records: int | None = None,
    ) -> float:
        """Return normalized confidence for a cluster in [0, 1].

        The normalizer is num_input_records * num_physical_ops_under_cluster.
        Confidence is based on the ratio of samples taken to the maximum
        possible (assuming all operators could be sampled for every record).
        """
        input_record_count = (
            self.max_input_records if max_input_records is None else max_input_records
        )
        max_possible_samples = max(input_record_count * len(cluster.physical_ops), 1)
        sample_count = self.cluster_sample_counts[id(cluster)]
        return min(sample_count / max_possible_samples, 1.0)

    def cluster_confidence_entropy(
        self,
        cluster: PhysicalOperatorCluster,
        max_input_records: int | None = None,
    ) -> float:
        """
        Return a confidence score which is based on the entropy of results obtained from the cluster.
        Clusters with a low entropy (i.e., more consistent results) will have a higher confidence score, while clusters with a high entropy (i.e., more varied results) will have a lower confidence score.
        The normalizer is num_input_records * num_physical_ops_under_cluster.

        NOTE: This method is not fully implemented. The breakpoint below marks
        the point where entropy calculation should be added. Currently it
        returns 0.0.
        """

        input_record_count = (
            self.max_input_records if max_input_records is None else max_input_records
        )
        max_possible_samples = max(input_record_count * len(cluster.physical_ops), 1)
        entropy = 0.0

        breakpoint()
        score = entropy

        return score

    def _cluster_has_available_physical_op(
        self,
        cluster: PhysicalOperatorCluster,
        record_id: str | int,
    ) -> bool:
        """Return whether a cluster has at least one unexecuted op for a record."""
        executed_physical_op_ids = self.executed_physical_op_ids_by_record[record_id]
        return any(
            op.get_full_op_id() not in executed_physical_op_ids
            for op in cluster.physical_ops
        )

    def _policy_scores(
        self,
        clusters: list[PhysicalOperatorCluster],
    ) -> dict[int, float]:
        """Normalize sibling cluster predictions according to the active policy.

        MaxQuality treats higher quality as better. MinCost and
        MinTime invert their metric so lower values receive higher scores.
        Normalization maps the best sibling to 1.0 and the worst to 0.0; if all
        siblings have the same value, all receive 1.0.
        """
        if not isinstance(self.policy, (MaxQuality, MinCost, MinTime)):
            raise NotImplementedError(
                f"Unsupported physical operator selection policy: {type(self.policy).__name__}"
            )

        cluster_values = {}
        for cluster in clusters:
            if cluster.cost_estimates is None:
                cluster_values[id(cluster)] = 0.0
            elif isinstance(self.policy, MaxQuality):
                cluster_values[id(cluster)] = cluster.cost_estimates.quality
            elif isinstance(self.policy, MinCost):
                cluster_values[id(cluster)] = cluster.cost_estimates.cost_per_record
            elif isinstance(self.policy, MinTime):
                cluster_values[id(cluster)] = cluster.cost_estimates.time_per_record

        min_value = min(cluster_values.values())
        max_value = max(cluster_values.values())
        if min_value == max_value:
            return {cluster_id: 1.0 for cluster_id in cluster_values}

        if isinstance(self.policy, MaxQuality):
            return {
                cluster_id: (value - min_value) / (max_value - min_value)
                for cluster_id, value in cluster_values.items()
            }

        return {
            cluster_id: (max_value - value) / (max_value - min_value)
            for cluster_id, value in cluster_values.items()
        }

    def _weighted_choice(
        self,
        clusters: list[PhysicalOperatorCluster],
        weights: list[float],
    ) -> PhysicalOperatorCluster:
        """Sample a cluster proportionally to non-negative weights."""
        total_weight = sum(weights)
        if total_weight <= 0:
            return self.rng.choice(clusters)

        threshold = self.rng.random() * total_weight
        running_weight = 0.0
        for cluster, weight in zip(clusters, weights, strict=True):
            running_weight += weight
            if running_weight >= threshold:
                return cluster

        return clusters[-1]

    def _average_cost_estimates(
        self,
        cost_estimates: list[OperatorCostEstimates],
    ) -> OperatorCostEstimates | None:
        """Average every populated OperatorCostEstimates field independently.

        Fields that are None in all estimates are left as None in the
        result.  If the list is empty, returns None.
        """
        if len(cost_estimates) == 0:
            return None

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

    def sample_physical_operator_cluster(
        self,
        cluster: PhysicalOperatorCluster,
        input_records: list[DataRecord] | list[tuple[DataRecord, DataRecord]],
        input_estimates: list[OperatorCostEstimates],
        sample_budget: int,
        optimization_stats: SentinelPlanStats,
        validator: Validator | None = None,
        on_sample_completed: Callable[[ProgressEvent], None] | None = None,
    ) -> tuple[List[DataRecord], OperatorCostEstimates] | None:
        """Sample each logical operator cluster for the configured budget.
        At the end of this method, each physical operator cluster will have
        been updated with cost estimates based on the observed sample statistics.

        This method perform exploration/exploitation over every
        logical operator cluster in topological order, from root logical operator to leaves.
        Each selected physical operator is executed on an actual sampled input record and the observed
        runtime/cost statistics are recorded back into the selected cluster path.
        """

        logical_source_op_ids = source_op_ids[logical_op_id]

        # progress_logical_op_id = (
        # progress_logical_op_ids.get(logical_op_id)
        # if progress_logical_op_ids is not None
        # else None
        # )

        logical_op = cluster.logical_op
        logical_op_id = logical_op.logical_op_id
        # input_payloads: dict[str, DataRecord | tuple[DataRecord, DataRecord] | list[DataRecord] | None]
        if isinstance(logical_op, BaseScan):
            input_payloads = {
                record_idx: record_idx
                for record_idx in range(len(logical_op.datasource))  # type: ignore
            }
        elif isinstance(logical_op, ContextScan):
            input_payloads = {f"{logical_op_id}:context": None}
        elif isinstance(logical_op, LogicalJoinOp):
            assert all(
                [len(x) == 2 for x in input_records]
            ), "Input records for join must be pairs of DataRecords"
            left_records, right_records = zip(*input_records)
            input_payloads = {}
            pair_idx = 0
            for left_record in left_records:
                for right_record in right_records:
                    input_payloads[
                        f"{left_record._id}:{right_record._id}:{pair_idx}"
                    ] = (left_record, right_record)
                    pair_idx += 1
        elif any(isinstance(op, AggregateOp) for op in cluster.physical_ops):
            input_payloads = (
                {f"{logical_op_id}:aggregate": input_records}
                if len(input_records) > 0
                else {}
            )
        elif len(logical_source_op_ids) == 1:
            input_payloads = {
                f"{record._id}:{record_idx}": record
                for record_idx, record in enumerate(input_records)
            }
        else:
            input_payloads = {}

        output_records = []
        # if len(input_payloads) == 0:
        # if progress_manager is not None and progress_logical_op_id is not None:
        # self.update_sampling_progress_total(
        # progress_manager,
        # progress_logical_op_id,
        # 0,
        # progress_total,
        # op_sample_budget,
        # )
        # progress_total -= op_sample_budget
        # continue

        input_record_ids = list(input_payloads.keys())
        budget = 0
        samples_drawn = 0
        self.total_sampling_rounds = sample_budget

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while budget < sample_budget:
                selections: list[tuple[int, PhysicalOperatorSelection]] = []
                while budget < sample_budget and len(selections) < self.max_workers:
                    try:
                        selection = self.select_next_sample(
                            cluster,
                            input_record_ids,  # type: ignore
                            budget,
                        )
                    except ValueError:
                        budget = sample_budget
                        break

                    self.executed_physical_op_ids_by_record[selection.record_id].add(
                        selection.physical_op.get_full_op_id()
                    )
                    selections.append((budget, selection))
                    budget += 1

                if len(selections) == 0:
                    break

                futures = {
                    executor.submit(
                        self.execute_and_score_physical_operator_sample,
                        logical_op,
                        selection.physical_op,
                        input_payloads[selection.record_id],  # type: ignore
                        validator,
                    ): (round_idx, selection)
                    for round_idx, selection in selections
                }
                sample_results = []
                for future in as_completed(futures):
                    round_idx, selection = futures[future]
                    record_set, input_count, elapsed_time, validation_gen_stats = (
                        future.result()
                    )
                    sample_results.append(
                        (
                            round_idx,
                            selection,
                            record_set,
                            input_count,
                            elapsed_time,
                            validation_gen_stats,
                        )
                    )

                for (
                    _,
                    selection,
                    record_set,
                    input_count,
                    elapsed_time,
                    validation_gen_stats,
                ) in sorted(
                    sample_results,
                    key=lambda result: result[0],
                ):
                    if input_count == 0:
                        continue

                    if optimization_stats is not None:
                        optimization_stats.add_record_op_stats(
                            cluster.logical_op.logical_op_id,  # type: ignore
                            record_set.record_op_stats,
                        )
                        optimization_stats.add_validation_gen_stats(
                            cluster.logical_op.logical_op_id,  # type: ignore
                            validation_gen_stats,
                        )

                    physical_op_id = selection.physical_op.get_full_op_id()
                    naive_cost_estimates = selection.cluster_path[-1].cost_estimates
                    if naive_cost_estimates is None:
                        raise ValueError(
                            f"Leaf cluster for physical op {physical_op_id} has no cost estimates"
                        )
                    estimate_cardinality_from_sample = isinstance(
                        selection.physical_op,
                        (
                            ContextScanOp,
                            ConvertOp,
                            FilterOp,
                            PhysicalJoinOp,
                            ScanPhysicalOp,
                        ),
                    )
                    observed_cost_estimates = self.estimate_sample_cost(
                        record_set,
                        input_count,
                        input_estimates,
                        naive_cost_estimates,
                        elapsed_time,
                        estimate_cardinality_from_sample=estimate_cardinality_from_sample,
                    )
                    self.record_execution(
                        selection,
                        observed_cost_estimates,
                    )
                    passed_records = [
                        record
                        for record in record_set.data_records
                        if record._passed_operator
                    ]
                    output_records.extend(passed_records)

                    if on_sample_completed is not None:
                        on_sample_completed(
                            ProgressEvent(
                                logical_op_id=cluster.logical_op.logical_op_id,  # type: ignore
                                output_count=len(passed_records),
                                incremental_cost=sum(
                                    stats.cost_per_record
                                    for stats in record_set.record_op_stats
                                )
                                + validation_gen_stats.cost_per_record,
                            )
                        )

                    # if (
                    #     progress_manager is not None
                    #     and progress_logical_op_id is not None
                    # ):
                    #     progress_cost = sum(
                    #         stats.cost_per_record
                    #         for stats in record_set.record_op_stats
                    #     )
                    #     progress_cost += validation_gen_stats.cost_per_record
                    #     progress_manager.incr(
                    #         progress_logical_op_id,
                    #         1,
                    #         display_text=(
                    #             f"{selection.physical_op.op_name()} "
                    #             f"({len(passed_records)} outputs)"
                    #         ),
                    #         total_cost=progress_cost,
                    #     )
                    samples_drawn += 1

        # if samples_drawn < sample_budget:
        #     if progress_manager is not None and progress_logical_op_id is not None:
        #         self.update_sampling_progress_total(
        #             progress_manager,
        #             progress_logical_op_id,
        #             samples_drawn,
        #             progress_total,
        #             sample_budget,
        #         )
        #         progress_total -= sample_budget - samples_drawn

    def execute_and_score_physical_operator_sample(
        self,
        logical_op,
        physical_op: PhysicalOperator,
        input_payload,
        validator: Validator | None,
    ) -> tuple[DataRecordSet, int, float, GenerationStats]:
        """Execute one optimizer sample and score it with the validator if present.

        This wrapper is necessary to allow the sample execution and scoring to be run in a thread pool executor, since the scoring may involve I/O or other blocking operations.
        """

        record_set, input_count, elapsed_time = self.execute_physical_operator_sample(
            logical_op,
            physical_op,
            input_payload,
        )
        validation_gen_stats = self.score_sample_quality(
            validator,
            physical_op,
            record_set,
        )
        return record_set, input_count, elapsed_time, validation_gen_stats

    def execute_physical_operator_sample(
        self,
        logical_op,
        physical_op: PhysicalOperator,
        input_payload,
    ) -> tuple[DataRecordSet, int, float]:
        """Execute a copied physical operator on one sampled optimizer input."""
        execution_op = physical_op.copy()
        if isinstance(execution_op, PhysicalJoinOp):
            execution_op._left_input_records = []
            execution_op._right_input_records = []
            execution_op._left_joined_record_ids = set()
            execution_op._right_joined_record_ids = set()
            execution_op.join_idx = 0
            execution_op.finished = False
        if isinstance(execution_op, BatchedOperator):
            execution_op.flushed = False
            execution_op._buffer = []

        if isinstance(execution_op, DistinctOp):
            execution_op._distinct_seen = set()

        start_time = time.time()
        if isinstance(logical_op, BaseScan):
            record_set = execution_op(input_payload)  # type: ignore
            input_count = 1
            record_set.input = input_payload
        elif isinstance(logical_op, ContextScan):
            raise NotImplementedError("ContextScan not yet supported in clustering")
            record_set = execution_op()
            input_count = 1
            record_set.input = input_payload
        elif isinstance(execution_op, PhysicalJoinOp):
            left_record, right_record = input_payload
            record_set, input_count = execution_op([left_record], [right_record])
            record_set.input = ([left_record], [right_record])
        elif isinstance(execution_op, AggregateOp):
            input_records = (
                input_payload if isinstance(input_payload, list) else [input_payload]
            )
            record_set = execution_op(candidates=input_records)
            input_count = len(input_records)
            record_set.input = input_records
        else:
            record_set = execution_op(input_payload)
            input_count = 1
            if (
                isinstance(execution_op, BatchedOperator)
                and len(record_set) == 0
                and execution_op.has_pending_batch()
            ):
                record_set = execution_op.flush()
            if record_set.input is None:
                record_set.input = input_payload

        execution_time = time.time() - start_time
        return record_set, input_count, execution_time

    def score_sample_quality(
        self,
        validator: Validator | None,
        physical_op: PhysicalOperator,
        record_set: DataRecordSet,
    ) -> GenerationStats:
        """Populate sampled record qualities using the provided validator."""
        if len(record_set.record_op_stats) == 0:
            return GenerationStats()

        if not isinstance(physical_op, (LLMConvert, LLMFilter, TopKOp, PhysicalJoinOp)):
            for record_op_stats in record_set.record_op_stats:
                record_op_stats.quality = 1.0
            return GenerationStats()

        if validator is None:
            return GenerationStats()

        if isinstance(physical_op, LLMConvert):
            if len(record_set.data_records) == 0:
                return GenerationStats()
            fields = physical_op.generated_fields
            input_record: DataRecord = record_set.input
            if physical_op.cardinality is Cardinality.ONE_TO_ONE:
                output = record_set.data_records[0].to_dict(project_cols=fields)
                output_str = record_set.data_records[0].to_json_str(
                    project_cols=fields,
                    bytes_to_str=True,
                    sorted=True,
                )
                full_hash = f"{hash(input_record)}{hash(output_str)}"
                score, validation_gen_stats, _ = validator._score_map(
                    physical_op,
                    fields,
                    input_record,
                    output,
                    full_hash,
                )
                record_set.record_op_stats[0].quality = score
                return validation_gen_stats
            else:
                output = [
                    data_record.to_dict(project_cols=fields)
                    for data_record in record_set.data_records
                ]
                output_strs = [
                    data_record.to_json_str(
                        project_cols=fields,
                        bytes_to_str=True,
                        sorted=True,
                    )
                    for data_record in record_set.data_records
                ]
                full_hash = f"{hash(input_record)}{hash(tuple(sorted(output_strs)))}"
                score, validation_gen_stats, _ = validator._score_flat_map(
                    physical_op,
                    fields,
                    input_record,
                    output,
                    full_hash,
                )
                for record_op_stats in record_set.record_op_stats:
                    record_op_stats.quality = score
                return validation_gen_stats

        if isinstance(physical_op, TopKOp):
            if len(record_set.data_records) == 0:
                return GenerationStats()
            fields = physical_op.generated_fields
            input_record: DataRecord = record_set.input
            output = record_set.data_records[0].to_dict(project_cols=fields)
            output_str = record_set.data_records[0].to_json_str(
                project_cols=fields,
                bytes_to_str=True,
                sorted=True,
            )
            full_hash = f"{hash(input_record)}{hash(output_str)}"
            score, validation_gen_stats, _ = validator._score_topk(
                physical_op,
                fields,
                input_record,
                output,
                full_hash,
            )
            record_set.record_op_stats[0].quality = score
            return validation_gen_stats

        if isinstance(physical_op, LLMFilter):
            validation_gen_stats = GenerationStats()
            scoring_op = physical_op
            filter_str = scoring_op.filter_obj.filter_condition

            input_records = (
                record_set.input
                if isinstance(record_set.input, list)
                else [record_set.input]
            )
            for input_record, data_record, record_op_stats in zip(
                input_records,
                record_set.data_records,
                record_set.record_op_stats,
                strict=True,
            ):
                output = data_record._passed_operator
                full_hash = f"{filter_str}{hash(input_record)}"
                score, sample_validation_gen_stats, _ = validator._score_filter(
                    scoring_op,
                    filter_str,
                    input_record,
                    output,
                    full_hash,
                )
                validation_gen_stats += sample_validation_gen_stats
                record_op_stats.quality = score
            return validation_gen_stats

        if isinstance(physical_op, PhysicalJoinOp):
            validation_gen_stats = GenerationStats()
            condition = physical_op.condition
            left_records, right_records = record_set.input
            record_idx = 0
            for left_record in left_records:
                for right_record in right_records:
                    data_record = record_set.data_records[record_idx]
                    output = data_record._passed_operator
                    full_hash = f"{condition}{hash(left_record)}{hash(right_record)}"
                    score, sample_validation_gen_stats, _ = validator._score_join(
                        physical_op,
                        condition,
                        left_record,
                        right_record,
                        output,
                        full_hash,
                    )
                    validation_gen_stats += sample_validation_gen_stats
                    record_set.record_op_stats[record_idx].quality = score
                    record_idx += 1
            return validation_gen_stats

        return GenerationStats()

    def estimate_sample_cost(
        self,
        record_set: DataRecordSet,
        input_count: int,
        source_cost_estimates: list[OperatorCostEstimates],
        naive_cost_estimates: OperatorCostEstimates,
        elapsed_time: float,
        estimate_cardinality_from_sample: bool = True,
    ) -> OperatorCostEstimates:
        """Convert sampled execution stats into operator cost estimates."""
        input_count = max(input_count, 1)
        record_op_stats = record_set.record_op_stats
        total_time = sum(stats.time_per_record for stats in record_op_stats)
        total_cost = sum(stats.cost_per_record for stats in record_op_stats)
        output_count = sum(
            record._passed_operator for record in record_set.data_records
        )

        input_cardinality = source_cost_estimates[0].cardinality
        if len(source_cost_estimates) > 1:
            right_source_cost_estimates = source_cost_estimates[1]
            input_cardinality *= right_source_cost_estimates.cardinality

        cardinality = naive_cost_estimates.cardinality
        if estimate_cardinality_from_sample:
            cardinality = (output_count / input_count) * input_cardinality

        observed_qualities = [
            stats.quality for stats in record_op_stats if stats.quality is not None
        ]
        quality = (
            sum(observed_qualities) / len(observed_qualities)
            if len(observed_qualities) > 0
            else naive_cost_estimates.quality
        )

        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=(
                total_time / input_count
                if len(record_op_stats) > 0
                else elapsed_time / input_count
            ),
            cost_per_record=(
                total_cost / input_count
                if len(record_op_stats) > 0
                else naive_cost_estimates.cost_per_record
            ),
            quality=quality,
        )

    def select_physical_operators(
        self,
        logical_plan: LogicalPlan,
        op_clusters: dict[str, PhysicalOperatorCluster],
        physical_operator_selector: PhysicalOperatorSampler,
    ) -> tuple[
        dict[str, PhysicalOperator],
        dict[str, OperatorCostEstimates],
    ]:
        """Select one concrete physical operator for every logical operator."""
        selected_physical_ops = {}
        selected_op_cost_estimates = {}
        for logical_op_id in logical_plan.topological_order:
            physical_op, cost_estimates = (
                physical_operator_selector.select_best_physical_operator(
                    op_clusters[logical_op_id]
                )
            )
            selected_physical_ops[logical_op_id] = physical_op
            selected_op_cost_estimates[logical_op_id] = cost_estimates

        return selected_physical_ops, selected_op_cost_estimates

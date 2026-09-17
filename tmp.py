from pydantic import BaseModel

from palimpzest.core.elements.filters import Filter
from palimpzest.query.operators.logical import BaseScan, FilteredScan, JoinOp
from palimpzest.query.optimizer.cluster.logical_optimizer import LogicalPlan


class Record(BaseModel):
    value: int


class FakeDataset:
    def __init__(self, dataset_id):
        self.id = dataset_id

    def __len__(self):
        return 1


def op_id(op):
    return op.get_logical_op_id()


A = BaseScan(FakeDataset("A"), output_schema=Record)
B = BaseScan(FakeDataset("B"), output_schema=Record)
C = FilteredScan(Filter(filter_condition="value > 0"), input_schema=Record, output_schema=Record)
D = FilteredScan(Filter(filter_condition="value < 0"), input_schema=Record, output_schema=Record)
E = FilteredScan(Filter(filter_condition="value == 0"), input_schema=Record, output_schema=Record)
F = JoinOp("D.value == E.value", input_schema=Record, output_schema=Record)
G = JoinOp("C.value == F.value", input_schema=Record, output_schema=Record)


A.logical_op_id = "A"
B.logical_op_id = "B"
C.logical_op_id = "C"
D.logical_op_id = "D"
E.logical_op_id = "E"
F.logical_op_id = "F"
G.logical_op_id = "G"



operators = {op_id(op): op for op in [A, B, C, D, E, F, G]}
edges = {
    op_id(A): [op_id(C), op_id(D)],
    op_id(B): [op_id(E)],
    op_id(D): [op_id(F)],
    op_id(E): [op_id(F)],
    op_id(C): [op_id(G)],
    op_id(F): [op_id(G)],
}

plan = LogicalPlan(operators, edges, root_op_id=op_id(G))
print(plan)
print(plan.topological_order)
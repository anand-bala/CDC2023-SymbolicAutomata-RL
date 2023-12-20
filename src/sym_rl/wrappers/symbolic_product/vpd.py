from typing import Mapping, Tuple, Union

import numpy as np

from syma.constraint.constraint import Constraint
from syma.constraint.node import (
    EQ,
    GEQ,
    GT,
    LEQ,
    LT,
    NEQ,
    And,
    BoolConst,
    BoolVar,
    IntConst,
    IntVar,
    NodeType,
    NodeVisitor,
    Not,
    Or,
    RealConst,
    RealVar,
)
from syma.constraint.node.node import Node
from syma.constraint.node.visitor import NodeVisitor

Value = Union[bool, int, float]
Values = Mapping[str, Value]


class _ValuePredicateDistance(NodeVisitor):
    """Value-Predicate Distance

    This function assumes that the semiring being used is the min-plus semiring, and the
    distance is the standard Hausdorff distance between the set of size 1 with the given
    valuation, and the set of valuations satisfying the predicate.

    NOTE: VPD is always positive
    """

    def __init__(self, values: Values, guard: Constraint) -> None:
        super().__init__()
        self.values = values
        self.guard = guard

    def __call__(self) -> Value:
        dist = self.visit(self.guard.formula)
        return dist

    def visit(self, node: Node) -> Value:
        return super().visit(node)

    def visitComparison(
        self, node: Union[GEQ, GT, LEQ, LT, EQ, NEQ]
    ) -> Tuple[Value, Value, str]:
        lhs = node.children[0]
        rhs = node.children[1]
        assert (lhs.node_type == NodeType.NumVar) or (rhs.node_type == NodeType.NumVar)
        if lhs.node_type == NodeType.NumVar:
            assert isinstance(lhs, (IntVar, RealVar))
            assert isinstance(rhs, (IntConst, RealConst))

            if isinstance(lhs, IntVar):
                comp_type = "int"
            else:
                comp_type = "real"
        else:
            assert isinstance(rhs, (IntVar, RealVar))
            assert isinstance(lhs, (IntConst, RealConst))

            if isinstance(rhs, IntVar):
                comp_type = "int"
            else:
                comp_type = "real"

        lhs_val = self.visit(lhs)
        rhs_val = self.visit(rhs)

        return lhs_val, rhs_val, comp_type

    def visitBoolConst(self, node: BoolConst) -> Value:
        if node.value:
            return 0
        return np.inf

    def visitIntConst(self, node: IntConst) -> Value:
        return int(node.value)

    def visitRealConst(self, node: RealConst) -> Value:
        return float(node.value)

    def visitBoolVar(self, node: BoolVar) -> Value:
        return bool(self.values[node.name])

    def visitIntVar(self, node: IntVar) -> Value:
        value = int(self.values[node.name])
        domain = self.guard._alphabet.get_domain(node.name)
        if not domain.is_empty():
            min_val, max_val = domain.as_tuple()
            return min(max(value, min_val), max_val)
        return value

    def visitRealVar(self, node: RealVar) -> Value:
        value = float(self.values[node.name])
        domain = self.guard._alphabet.get_domain(node.name)
        if not domain.is_empty():
            min_val, max_val = domain.as_tuple()
            return min(max(value, min_val), max_val)
        return value

    def visitGEQ(self, node: GEQ) -> Value:
        lhs, rhs, _ = self.visitComparison(node)
        if lhs >= rhs:
            return 0
        else:
            return rhs - lhs

    def visitGT(self, node: GT) -> Value:
        lhs, rhs, comp_type = self.visitComparison(node)
        if lhs > rhs:
            return 0
        else:
            addition = 1 if comp_type == "int" else 0
            return rhs - lhs + addition

    def visitLEQ(self, node: LEQ) -> Value:
        lhs, rhs, _ = self.visitComparison(node)
        if lhs <= rhs:
            return 0
        else:
            return lhs - rhs

    def visitLT(self, node: LT) -> Value:
        lhs, rhs, comp_type = self.visitComparison(node)
        if lhs < rhs:
            return 0
        else:
            addition = 1 if comp_type == "int" else 0
            return lhs - rhs + addition

    def visitEQ(self, node: EQ) -> Value:
        lhs, rhs, _ = self.visitComparison(node)
        if lhs == rhs:
            return 0
        else:
            return abs(lhs - rhs)

    def visitNEQ(self, node: NEQ) -> Value:
        lhs, rhs, comp_type = self.visitComparison(node)
        if lhs == rhs:
            return 0
        else:
            addition = 1 if comp_type == "int" else 0
            return abs(lhs - rhs) + addition

    def visitNot(self, _: Not) -> Value:
        raise ValueError("Expression needs to be in DNF. `Not` should not exist")

    def visitAnd(self, node: And) -> Value:
        children = [self.visit(child) for child in node.children]
        return sum(children)

    def visitOr(self, node: Or) -> Value:
        children = [self.visit(child) for child in node.children]
        return min(children)


def value_predicate_distance(values: Values, guard: Constraint) -> Value:
    vpd = _ValuePredicateDistance(values, guard)
    return vpd()

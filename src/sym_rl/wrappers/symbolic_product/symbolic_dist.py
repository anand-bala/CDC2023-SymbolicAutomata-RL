from collections import deque
from itertools import product
from typing import Deque, Dict, List, Tuple, Union

import numpy as np
from syma.alphabet import Interval
from syma.automaton import SymbolicAutomaton

from syma.constraint.constraint import Constraint
from syma.constraint.node.node import (EQ, GEQ, GT, LEQ, LT, NEQ, And,
                                       BoolConst, ComparisonOp, IntConst,
                                       IntVar, Node, NodeType, Or, RealConst,
                                       RealVar)

Value = Union[bool, int, float]
NumConst = Union[IntConst, RealConst]
NumVar = Union[IntVar, RealVar]
ComparisonNode = Union[EQ, GEQ, GT, LEQ, LT, NEQ]


def _cmp_to_interval(expr: ComparisonNode) -> List[Interval]:
    # Assume the node is in canonical form.
    x: NumVar = expr.children[0]  # type: ignore
    c: NumConst = expr.children[1]  # type: ignore
    op = expr.op_type

    open_op = op in (ComparisonOp.GT, ComparisonOp.LT, ComparisonOp.NEQ)

    # Assume that all float comparisons do not deal with LT, GT, or NEQ
    offset = 1 if x.is_int() and open_op else 0

    if op in (ComparisonOp.LT, ComparisonOp.LEQ):
        return [Interval(-np.inf, c.value - offset)]
    elif op in (ComparisonOp.GEQ, ComparisonOp.GT):
        return [Interval(c.value + offset, np.inf)]
    elif op == ComparisonOp.EQ:
        return [Interval(c.value, c.value)]
    elif op == ComparisonOp.NEQ:
        if x.is_int():
            return [
                Interval(-np.inf, c.value - offset),
                Interval(c.value + offset, np.inf),
            ]
        else:
            return [Interval(-np.inf, np.inf)]
    else:
        raise RuntimeError("Unreachable")


class _PredicateDistance(object):
    """
    Given two predicates, compute the symbolic distances between the two convex sets
    determined by the predicates
    """

    # NOTE: This class isn't a NodeVisitor as it is technically not visiting 1 node.
    # It is visiting 2 nodes.

    def __init__(self, expr1: Constraint, expr2: Constraint):
        assert expr1.alphabet == expr2.alphabet

        self.alphabet = expr1.alphabet
        self.expr1 = expr1.to_dnf()
        self.expr2 = expr2.to_dnf()

    def evaluate(self) -> float:
        return self._eval_trampoline(self.expr1.formula, self.expr2.formula)

    def _eval_trampoline(self, a: Node, b: Node) -> float:
        node_types = a.node_type, b.node_type

        if b.node_type == NodeType.BoolConst:
            assert isinstance(b, BoolConst)
            return self._to_bool_dist(a, b)
        if a.node_type == NodeType.BoolConst:
            assert isinstance(a, BoolConst)
            return self._to_bool_dist(b, a)

        # If b is a Comparison operation
        if node_types == (NodeType.Comparison, NodeType.Comparison):
            assert isinstance(a, (LT, LEQ, GT, GEQ, NEQ, EQ))
            assert isinstance(b, (LT, LEQ, GT, GEQ, NEQ, EQ))
            return self._predicate_predicate_dist(a, b)
        elif node_types == (NodeType.And, NodeType.Comparison):
            assert isinstance(a, And)
            assert isinstance(b, (LT, LEQ, GT, GEQ, NEQ, EQ))
            return self._to_cnf_dist(b, a)
        elif node_types == (NodeType.Comparison, NodeType.And):
            assert isinstance(b, And)
            assert isinstance(a, (LT, LEQ, GT, GEQ, NEQ, EQ))
            return self._to_cnf_dist(a, b)
        elif node_types == (NodeType.Or, NodeType.Comparison):
            assert isinstance(a, Or)
            assert isinstance(b, (LT, LEQ, GT, GEQ, NEQ, EQ))
            return self._to_dnf_dist(b, a)
        elif node_types == (NodeType.Comparison, NodeType.Or):
            assert isinstance(b, Or)
            assert isinstance(a, (LT, LEQ, GT, GEQ, NEQ, EQ))
            return self._to_dnf_dist(a, b)
        elif node_types == (NodeType.And, NodeType.And):
            assert isinstance(a, And)
            assert isinstance(b, And)
            return self._to_cnf_dist(a, b)
        elif node_types == (NodeType.And, NodeType.Or):
            assert isinstance(a, And)
            assert isinstance(b, Or)
            return self._to_dnf_dist(a, b)
        elif node_types == (NodeType.Or, NodeType.And):
            assert isinstance(a, Or)
            assert isinstance(b, And)
            return self._to_dnf_dist(b, a)
        elif node_types == (NodeType.Or, NodeType.Or):
            assert isinstance(a, Or)
            assert isinstance(b, Or)
            return self._to_dnf_dist(a, b)

        # if b.node_type == NodeType.And:
        #     assert isinstance(b, And)
        #     return self._to_cnf_dist(a, b)
        # elif b.node_type == NodeType.Or:
        #     assert isinstance(b, Or)
        #     return self._to_dnf_dist(a, b)
        else:
            raise TypeError(
                f"Cannot find distance between unsupported types: {repr(a)}, {repr(b)}"
            )

    def _to_bool_dist(self, a: Node, b: BoolConst) -> float:
        if a.node_type == NodeType.BoolConst:
            assert isinstance(a, BoolConst)
            if a.value == b.value:
                return 0

        if b.value is True:
            return 0

        raise ValueError(
            f"`False` shouldn't exist in these expressions:\n\t{self.expr1}\n\t{self.expr2}"
        )

    def _predicate_predicate_dist(self, a: ComparisonNode, b: ComparisonNode) -> float:

        op_a = a.op_type
        # assert op_a not in (
        #     ComparisonOp.NEQ,
        #     ComparisonOp.EQ,
        # ), "EQ and NEQ operations should have been removed using to_nnf"
        assert isinstance(a.children[0], (IntVar, RealVar))
        assert isinstance(a.children[1], (IntConst, RealConst))
        x_a: NumVar = a.children[0]
        c_a: NumConst = a.children[1]

        op_b = b.op_type
        # assert op_b not in (
        #     ComparisonOp.NEQ,
        #     ComparisonOp.EQ,
        # ), "EQ and NEQ operations should have been removed using to_nnf"
        assert isinstance(b.children[0], (IntVar, RealVar))
        assert isinstance(b.children[1], (IntConst, RealConst))
        x_b: NumVar = b.children[0]
        c_b: NumConst = b.children[1]

        if x_a.name != x_b.name:
            # Variables are not the same, so they are uncomparable.
            # HACK: Assume that they are as close as possible
            # UPDATE: Makes more sense to be pessimistic and make this inf
            # Then pray that another term in the expression has finite distance.
            return np.nan

        # Now, we know that the var names are the same.
        # We should get the closed intervals these predicates represent.
        set_a = _cmp_to_interval(a)
        set_b = _cmp_to_interval(b)

        # Check if the intervals overlap
        for i, j in product(set_a, set_b):
            if i.overlaps(j):
                return 0

        # Now we know that they are truly comparable, i.e., there is some separation
        # between them and all the intervals in set_a and set_b are disjoint.

        # Let's convert the set of intervals to a list of tuples, and sort them by
        # starting point.
        set_a = sorted((i.as_tuple() for i in set_a))
        set_b = sorted((i.as_tuple() for i in set_b))
        # Compute the minimum Hausdorff distance between any of the intervals
        dist = np.inf
        for i, j in product(set_a, set_b):
            if i < j:
                # i is to the left of j, so compare i.end to j.start
                dist = min(dist, abs(i[1] - j[0]))
            else:  # j < i
                dist = min(dist, abs(j[1] - i[0]))

        # # Get the absolute distance between the constants.
        # dist = abs(c_a.value - c_b.value)
        # if not x_a.is_int():
        #     # If the variable is not an int type, just return the distance as is.
        #     #
        #     # NOTE: This is because we can't take into the lack of boundaries for sets
        #     # created by LT and GT operations.
        #     return dist
        #
        # # If the variable is an integer, we need to check the operation type.
        # if op_a in (ComparisonOp.LT, ComparisonOp.GT):
        #     # Add 1 to the distance to take into account the shifted boundary
        #     dist += 1
        # if op_b == ComparisonOp.LT or op_b == ComparisonOp.GT:
        #     # Add 1 to the distance to take into account the shifted boundary
        #     dist += 1

        return dist

    def _to_cnf_dist(self, a: Node, b: And) -> float:
        # We will find the distance of each term in `b` to the node `a`. Then,
        # we will sum the distances (assuming min-plus semiring)
        dist = 0
        for term in b.children:
            d = self._eval_trampoline(term, a)
            if np.isnan(d):
                dist += 0
            else:
                dist += d
        return dist

    def _to_dnf_dist(self, a: Node, b: Or) -> float:
        # We will find the distance of each term in `b` to the node `a`. Then,
        # we will minimize the distances (assuming min-plus semiring)
        dist = np.inf
        for term in b.children:
            d = self._eval_trampoline(term, a)
            if np.isnan(d):
                continue
            else:
                dist = min(dist, d)

        return dist


def _predicate_distance(a: Constraint, b: Constraint) -> float:
    return _PredicateDistance(a, b).evaluate()


def symbolic_distance(
    aut: SymbolicAutomaton, target: int
) -> Dict[Tuple[int, int], float]:
    r"""
    Symbolic Distance is computed using the following recursive formulation:

    .. math::
       :nowrap:

        dist_{sym}(q, q') = \begin{cases}
            0 & \text{if}~ q' \in Acc \\
            \min_{(q', q'') \in E} (dist(\psi(q, q'), \psi(q', q'') + dist_{sym}(q', q''))) & \text{otherwise}.
        \end{cases}

    Due to the recursive nature of this value, we can use a simple dynamic programming
    algorithm to compute the `dist_{sym}`, starting with the base case of an accepting
    set.
    """

    # Create a queue to lazily add new nodes to relax, similar to the Shortest Path
    # Faster Algorithm:
    #
    #   https://en.wikipedia.org/wiki/Shortest_Path_Faster_Algorithm
    #
    queue: Deque[Tuple[int, int]] = deque()
    # Initialize the distances and add edges incident on accepting states to the queue
    dist: Dict[Tuple[int, int], float] = {e: np.inf for e in aut._graph.edges}
    for (src, dst) in aut._graph.edges:
        if dst == target:
            dist[(src, dst)] = 0
            queue.append((src, dst))

    # While the queue isn't empty, we will process each incoming edge to the
    # _predecessor_ of the popped edge from the queue.
    while len(queue) > 0:
        cur = queue.popleft()
        src, dst = cur
        # Disregard self loops as they do not contribute to progress in rewards.
        if src == dst:
            continue
        cur_psi = aut.get_guard(src, dst)
        for prev in aut.in_edges(src):
            prev_psi = aut.get_guard(*prev)
            # Get the distance of the current guard and this incoming edge guard.
            weight = _predicate_distance(prev_psi, cur_psi)
            assert weight >= 0

            # Check if this gets a better distance
            new_dist = weight + dist[cur]
            if new_dist < dist[prev]:
                dist[prev] = new_dist
                if prev not in queue:
                    queue.append(prev)

    return dist

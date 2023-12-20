from typing import Mapping, Protocol, TypeVar, Union, Dict

T = TypeVar("T", contravariant=True)


class ValuationFn(Protocol[T]):
    """A state valuation function.

    It takes a state from the MDP and outputs a mapping of variable names to values.
    """

    def __call__(self, observation: T, info: Dict) -> Mapping[str, Union[int, float]]:
        ...

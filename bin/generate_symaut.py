import argparse
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple

try:
    import spot
except ImportError as e:
    print(e)
    print("")
    print(
        """
The best way to install SPOT is to using conda:

    conda install spot -c conda-forge

If you are not using conda, see https://spot.lrde.epita.fr/install.html
"""
    )
    raise e

import rtamt
from rtamt.node.abstract_node import AbstractNode

from rtamt_helpers.to_automaton import print_automaton_gen_code
from rtamt_helpers.to_ltl_string import to_ltl_string


class Arguments(NamedTuple):
    formula: str
    output_file: Optional[Path]
    use_ltlf: bool


def parse_args() -> Arguments:
    parser = argparse.ArgumentParser(
        description="Convert a STL formula into a deterministic finite state automaton",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-o",
        "--output-file",
        help="Output the automaton as a labelled graph",
        type=lambda p: Path(p).resolve(),
        default=None,
    )

    parser.add_argument(
        "--use-ltlf",
        action="store_true",
        help="Use LTLf translation",
    )

    parser.add_argument("formula", help="Temporal Logic formula to parse", type=str)

    args = parser.parse_args()

    return Arguments(
        formula=args.formula, output_file=args.output_file, use_ltlf=args.use_ltlf
    )


def get_stl_formula(formula: str) -> rtamt.STLSpecification:
    spec = rtamt.STLSpecification()

    # TODO: The vars are hard coded for our experiments
    spec.declare_var("x", "int")
    spec.declare_var("y", "int")

    spec.spec = formula

    spec.parse()

    return spec


def get_ltl_formula(
    spec: rtamt.STLSpecification,
) -> Tuple[str, Dict[str, AbstractNode]]:
    return to_ltl_string(spec)


def main():
    args = parse_args()
    stl_spec = get_stl_formula(args.formula)
    print_automaton_gen_code(stl_spec, use_ltlf=args.use_ltlf)


if __name__ == "__main__":
    main()

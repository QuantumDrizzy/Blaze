"""Minimal CLI for Fase 1 (placeholder — real CLI in later phases)."""

import argparse
import sys

from blaze import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="blaze", description="Blaze Fase 1 prototype")
    parser.add_argument("--version", action="store_true", help="print version")
    parser.add_argument("--bench", action="store_true", help="run classical benchmark")
    parser.add_argument("--quantum", action="store_true", help="run quantum golden demo (needs cirq)")
    args = parser.parse_args(argv)

    if args.version:
        print(f"blaze {__version__} (Fase 1 Python prototype)")
        return 0

    if args.bench:
        from blaze.examples.classical_benchmark import main as bench_main
        bench_main()
        return 0

    if args.quantum:
        try:
            from blaze.examples.quantum_compression import main as qmain
            qmain()
        except Exception as e:
            print(f"Quantum demo requires cirq: {e}")
            return 2
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

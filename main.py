"""
main.py: run all experiments sequentially.
Usage:
    python main.py          # run all three experiments
    python main.py --exp 1  # run only experiment 1
    python main.py --exp 2
    python main.py --exp 3
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(
        description='Online Learning for Budget-Constrained Ad Auctions')
    parser.add_argument('--exp', type=int, choices=[1, 2, 3],
                        help='Which experiment to run (default: all)')
    args = parser.parse_args()

    if args.exp is None or args.exp == 1:
        print("\n" + "="*60)
        print(" EXPERIMENT 1: Single Campaign, Stochastic Environment")
        print("="*60)
        from experiments.exp_single_campaign import main as exp1
        exp1()

    if args.exp is None or args.exp == 2:
        print("\n" + "="*60)
        print(" EXPERIMENT 2: Multi-Campaign, Stochastic Environment")
        print("="*60)
        from experiments.exp_multi_campaign import main as exp2
        exp2()

    if args.exp is None or args.exp == 3:
        print("\n" + "="*60)
        print(" EXPERIMENT 3: Multi-Campaign, Non-Stationary Environment")
        print("="*60)
        from experiments.exp_non_stationary import main as exp3
        exp3()

    print("\n✓ All experiments completed. Results saved to results/")


if __name__ == '__main__':
    main()

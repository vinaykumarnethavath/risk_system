"""
main.py — Entry point for Corporate Intelligence AI platform.
Full pipeline will be integrated in later commits.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Corporate Integrity & Growth Intelligence AI"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock ticker symbol to analyse (default: AAPL)",
    )
    args = parser.parse_args()

    print(f"Corporate Intelligence AI — analysing {args.ticker}")
    print("Pipeline modules will be connected in upcoming commits.")


if __name__ == "__main__":
    main()

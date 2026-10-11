"""CLI: trubeat hr|train|predict|eval. (SRS §3)"""

import argparse


def main() -> None:
    p = argparse.ArgumentParser(prog="trubeat")
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("hr", help="estimate heart rate from a video")
    h.add_argument("video")

    t = sub.add_parser("train", help="train the detector on a manifest")
    t.add_argument("--manifest", required=True)
    t.add_argument("--config", required=True)

    pr = sub.add_parser("predict", help="classify one video (offline)")
    pr.add_argument("video")
    pr.add_argument("--model", required=True)

    e = sub.add_parser("eval", help="batch-score a dataset manifest")
    e.add_argument("--dataset", required=True)
    e.add_argument("--config", required=True)

    args = p.parse_args()
    raise NotImplementedError(f"command '{args.cmd}' not implemented yet")


if __name__ == "__main__":
    main()

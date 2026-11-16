"""Run the independent raw-observation Haiqal baseline."""

from __future__ import annotations

from pathlib import Path

from possession_lab.haiqal_baseline import counts, predict, write


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    labels = predict()
    output = ROOT / "evaluation" / "results" / "baseline_haiqal_raw_viterbi.csv"
    write(output, labels)
    print(counts(labels), {label: int((labels == label).sum()) for label in "HADX"})
    print(f"-> {output}")


if __name__ == "__main__":
    main()

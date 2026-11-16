"""Run the label-free broadcast evidence audit and save its reports."""

from pathlib import Path

from possession_lab.broadcast_alignment import (
    build_broadcast_audit,
    build_source_signatures,
    find_local_alignment_candidates,
    find_local_sequence_candidates,
    identify_tactical_need_windows,
    summarise_candidate_windows,
)
from possession_lab.temporal_rules import load_rules


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "evaluation" / "reports"


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    audit = build_broadcast_audit()
    windows = summarise_candidate_windows(audit)
    tactical_audit = build_broadcast_audit(source="tactical")
    rules = load_rules()
    broadcast_rules = rules["broadcast"]
    graded_ranges = tuple(tuple(pair) for pair in rules["competition"]["graded_ranges"])
    needs = identify_tactical_need_windows(
        tactical_audit,
        graded_ranges=graded_ranges,
        fps=float(broadcast_rules["fps"]),
    )
    tactical_signatures = build_source_signatures("tactical")
    broadcast_signatures = build_source_signatures("broadcast")
    matches = find_local_alignment_candidates(tactical_signatures, broadcast_signatures, needs)
    sequence_matches = find_local_sequence_candidates(
        tactical_signatures,
        broadcast_signatures,
        needs,
        sequence_blocks=15,
        expected_players=float(broadcast_rules["expected_players"]),
    )
    audit_path = REPORTS / "broadcast_evidence_blocks.csv"
    windows_path = REPORTS / "broadcast_candidate_windows.csv"
    needs_path = REPORTS / "tactical_broadcast_need_windows.csv"
    matches_path = REPORTS / "tactical_broadcast_alignment_candidates.csv"
    sequence_path = REPORTS / "tactical_broadcast_sequence_candidates.csv"
    audit.to_csv(audit_path, index=False)
    windows.to_csv(windows_path, index=False)
    needs.to_csv(needs_path, index=False)
    matches.to_csv(matches_path, index=False)
    sequence_matches.to_csv(sequence_path, index=False)
    print(f"blocks: {len(audit):,}")
    print(f"live-evidence blocks: {int(audit['live_evidence'].sum()):,}")
    print(f"candidate windows: {len(windows):,}")
    print(f"tactical need windows: {len(needs):,}")
    print(f"alignment candidates: {len(matches):,}")
    print(f"accepted diagnostic matches: {int(matches['candidate_accepted'].sum()) if not matches.empty else 0:,}")
    print(f"sequence candidates: {len(sequence_matches):,}")
    print(f"accepted sequence candidates: {int(sequence_matches['candidate_accepted'].sum()) if not sequence_matches.empty else 0:,}")
    print(f"reports: {audit_path}, {windows_path}, {needs_path}, {matches_path}, {sequence_path}")


if __name__ == "__main__":
    main()


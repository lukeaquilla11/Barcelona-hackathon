"""Public scenario API; legacy modules remain compatible for existing callers."""

from possession_lab.temporal_rules import *
from possession_lab.decoder_experiments import (
    add_scenario_episode_features,
    bounded_bidirectional_scenario_resolver,
    classify_scenarios,
    dead_ball_boundary_evidence,
    resolve_aligned_reception_switch,
    resolve_reception_switch_inside_interval,
    scenario_confidence_scores,
    scenario_gated_labels,
)


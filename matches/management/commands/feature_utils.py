# prediction/matches/feature_utils.py
import numpy as np

# Canonical numeric features that both train & predict use
CANONICAL_FEATS = {
    "h_gf10","a_gf10","d_gf10",
    "h_ga10","a_ga10","h_gd10","a_gd10",
    "h_sot10","a_sot10","d_sot10",
    "h_sot_pct10","a_sot_pct10",
    "h_conv10","a_conv10",
    "h_poss10","a_poss10",   # should already be stored as normalized 0–1
    "h_clean_sheets10","a_clean_sheets10",
    "h_corners_for10","a_corners_for10",
    "h_cards_for10","a_cards_for10",
    "h_rest_days","a_rest_days","d_rest_days",
    "h_matches_14d","a_matches_14d",
    "h_stats_missing","a_stats_missing",
}

def build_feature_vector(obj) -> tuple[dict, np.ndarray]:
    """
    Take either a Match or MLTrainingMatch row and return
    (dict, np.array) in canonical order.
    """
    feats = {f: getattr(obj, f, 0.0) if hasattr(obj, f) else 0.0
             for f in CANONICAL_FEATS}
    X = np.array([feats[f] for f in CANONICAL_FEATS], dtype=float)
    return feats, X

#!/usr/bin/env python3
"""Generate the SHAP beeswarm for the exact 33-feature pre-SAST design matrix."""
from pathlib import Path
import argparse, importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


def load_protocol(path: Path):
    spec = importlib.util.spec_from_file_location("lr_rf_protocol", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="case_model_problem_statement_features.csv")
    ap.add_argument("--protocol-script", required=True,
                    help="lr_rf_post_generation_30runs.py")
    ap.add_argument("--output", default="imgs/fig_shap_33_features.png")
    ap.add_argument("--max-rows", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mod = load_protocol(Path(args.protocol_script))
    df = pd.read_csv(args.input)
    mod.check_required_columns(df)
    work, X = mod.build_design_matrix(df)
    y = work[mod.TARGET].astype(int).to_numpy()

    med = X.median(numeric_only=True).fillna(0.0)
    Xf = X.fillna(med).fillna(0.0)

    rf = mod.make_rf(args.seed)
    rf.fit(Xf, y)

    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(Xf), size=min(args.max_rows, len(Xf)), replace=False)
    Xs = Xf.iloc[idx]

    explainer = shap.TreeExplainer(rf)
    values = explainer.shap_values(Xs)
    if isinstance(values, list):
        values = values[1]
    elif getattr(values, "ndim", 0) == 3:
        values = values[:, :, 1]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 7))
    shap.summary_plot(values, Xs, feature_names=X.columns.tolist(),
                      show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Generated {out} from {X.shape[1]} features and {len(df)} rows")


if __name__ == "__main__":
    main()

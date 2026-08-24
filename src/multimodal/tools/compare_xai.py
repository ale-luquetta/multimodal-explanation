#!/usr/bin/env python3
"""Compare unimodal and multimodal XAI outputs — ablation faithfulness.

Offline tool: reads the CSVs of two finished runs, pairs the screens they have
in common and compares visual faithfulness between the two pipelines. 

**Metric: delta_p_img** 

- ``delta_p_img = |p_orig - p_img|`` — how much the confidence moves when the
  top-K most salient regions are zeroed. Higher means those regions weighed
  more on the decision, i.e. a more faithful visual explanation.
- On the multimodal side the value is computed per (screen, review) pair and
  aggregated as the mean per screen. The unimodal side has no reviews, so it is
  the single value of the screen.

Outputs:

- ``comparison_per_screen.csv`` — one row per paired screen (present in both
  runs), with the delta_p_img of each pipeline.
- ``comparison_per_app.csv`` — mean delta_p_img per app. Here each side uses
  all screens of the app in its own run, not only the paired ones.
- ``comparison_summary.txt`` — descriptive stats of the per-screen delta_p_img
  (mean, median, how many screens each pipeline led on), over the whole set and
  split by class, plus the per-app table.

CLI entry: "Compare unimodal vs multimodal XAI (ΔP_img)".
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from multimodal.common.paths import RESULTS_EXPLANATIONS_COMPARISON


# =============================================================================
# LOADERS
# =============================================================================
def _iter_apps(run_dir: Path) -> dict[str, Path]:
    """Return {safe_pkg: app_dir} for a run laid out as ``apps/<pkg>/``."""
    apps_dir = run_dir / "apps"
    if not apps_dir.exists():
        raise FileNotFoundError(f"'apps' directory not found in {run_dir}")
    result = {}
    for d in sorted(apps_dir.iterdir()):
        if d.is_dir():
            result[d.name] = d
    return result


def _load_screen_csv(app_dir: Path) -> pd.DataFrame | None:
    """Load the ``screen_explanations.csv`` of one app."""
    sc_path = app_dir / "screen_explanations.csv"
    if not sc_path.exists():
        return None
    try:
        sc = pd.read_csv(sc_path)
    except Exception as e:
        print(f"[compare_xai] Failed to load {app_dir.name}: {e}")
        return None
    sc["screen_id"] = sc["screen_id"].astype(str)
    return sc


def _load_app_names(run_dir: Path) -> dict[str, str]:
    """Return {safe_pkg: app_name} read from ``screen_explanations.json``.

    The per-app CSVs do not carry the commercial name of the app, only the
    ``package_name`` (and the multimodal side not even that). The per-app table
    is identified by name, so it comes from here.
    """
    json_path = run_dir / "screen_explanations.json"
    if not json_path.exists():
        return {}
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[compare_xai] Failed to read {json_path}: {e}")
        return {}
    out: dict[str, str] = {}
    for scr in data:
        pkg = scr.get("package_name")
        name = scr.get("app_name")
        if pkg and name:
            out[str(pkg).replace(".", "_")] = str(name)
    return out


def _load_app_confidence(run_dir: Path) -> dict[str, float]:
    """Return {safe_pkg: app confidence}, used to order the per-app table.

    Reproduces ``app_confidence`` from ``app_explanations.json``: the mean of
    the screen predictions and, when that falls below 0.5, its complement.
    Uses all screens in ``screen_explanations.csv``, including those without
    delta_p_img — restricting to screens that have the metric changes the value
    for some apps and would reorder the table.
    """
    out: dict[str, float] = {}
    for safe_pkg, app_dir in _iter_apps(run_dir).items():
        sc = _load_screen_csv(app_dir)
        if sc is None or "prediction" not in sc.columns:
            continue
        preds = pd.to_numeric(sc["prediction"], errors="coerce").dropna()
        if preds.empty:
            continue
        score = float(preds.mean())
        out[safe_pkg] = score if score > 0.5 else 1.0 - score
    return out


def _load_per_review_delta_p(app_dir: Path) -> dict[str, float]:
    """Return {screen_id: mean delta_p_img over that screen's reviews}.

    This is the aggregation: on the multimodal side delta_p_img is computed 
    per (screen, review) pair, so the mean per screen is the unit comparable 
    to the unimodal per-screen value. Returns {} when ``per_review_analyses.csv`` 
    is absent, which is the unimodal case, since it has no reviews.
    """
    pr_path = app_dir / "per_review_analyses.csv"
    if not pr_path.exists():
        return {}
    try:
        pr = pd.read_csv(pr_path)
    except Exception as e:
        print(f"[compare_xai] Failed to load {pr_path}: {e}")
        return {}
    if "delta_p_img" not in pr.columns or "screen_id" not in pr.columns:
        return {}
    pr["screen_id"] = pr["screen_id"].astype(str)
    grouped = pr.groupby("screen_id")["delta_p_img"].mean()
    return {sid: float(v) for sid, v in grouped.items() if pd.notna(v)}


def _build_per_screen_table(run_dir: Path, label: str) -> pd.DataFrame:
    """Walk the run's apps and screens and return a long DataFrame with one row
    per (safe_pkg, screen_id) and columns prefixed by ``label``.

    Screens without delta_p_img are dropped: without the metric there is
    nothing to compare.
    """
    apps = _iter_apps(run_dir)
    n_pr_screens = 0
    n_skipped = 0
    rows: list[dict] = []
    for safe_pkg, app_dir in apps.items():
        sc = _load_screen_csv(app_dir)
        if sc is None:
            continue
        pr_map = _load_per_review_delta_p(app_dir)
        n_pr_screens += len(pr_map)
        for _, screen_row in sc.iterrows():
            sid = str(screen_row["screen_id"])
            delta_p = pr_map.get(sid)
            if delta_p is None:
                raw = screen_row.get("delta_p_img")
                delta_p = float(raw) if pd.notna(raw) else None
            if delta_p is None:
                n_skipped += 1
                continue
            rows.append({
                "safe_pkg": safe_pkg,
                "package_name": screen_row.get("package_name", ""),
                "screen_id": sid,
                "true_label": int(screen_row.get("true_label", 0)),
                "is_correct": bool(screen_row.get("is_correct", False)),
                "delta_p_img": delta_p,
                "prediction": float(screen_row.get("prediction") or 0.0),
                "confidence": float(screen_row.get("confidence") or 0.0),
            })
    # Where the values came from — guards against silently comparing different
    # aggregations.
    print(
        f"[compare_xai] {label}: {len(rows)} screens | "
        f"delta_p_img from review mean on {n_pr_screens} | "
        f"{n_skipped} without delta_p_img (dropped)"
    )
    df = pd.DataFrame(rows)
    df = df.add_prefix(f"{label}_")
    # Restore the merge keys without the prefix.
    df = df.rename(columns={
        f"{label}_safe_pkg": "safe_pkg",
        f"{label}_screen_id": "screen_id",
    })
    return df


# =============================================================================
# COMPARISON
# =============================================================================
def compare_pair(uni_df: pd.DataFrame, multi_df: pd.DataFrame) -> pd.DataFrame:
    """Merge on (safe_pkg, screen_id), keeping only screens present in both."""
    merged = uni_df.merge(multi_df, on=["safe_pkg", "screen_id"], how="inner")
    print(
        f"[compare_xai] Pairing: {len(merged)} screens in common "
        f"(uni={len(uni_df)}, multi={len(multi_df)})"
    )
    return merged


def build_per_app_table(
    uni_df: pd.DataFrame,
    multi_df: pd.DataFrame,
    app_names: dict[str, str],
    app_confidence: dict[str, float],
) -> pd.DataFrame:
    """Mean delta_p_img per app

    Each side is the mean over all screens of that app in its own run, not only
    the paired ones, which is the comparable number when the two runs cover
    slightly different sets of screens.

    Ordering: class (good before bad) and, within it, app confidence
    descending, coming from the selection of the most confident apps of each
    class.
    """
    def _agg(df: pd.DataFrame, label: str) -> pd.DataFrame:
        out = (
            df.groupby("safe_pkg")
            .agg(**{
                f"delta_p_img_{label}": (f"{label}_delta_p_img", "mean"),
                f"n_screens_{label}": (f"{label}_delta_p_img", "size"),
                f"true_label_{label}": (f"{label}_true_label", "first"),
            })
            .reset_index()
        )
        return out

    per_app = _agg(uni_df, "uni").merge(_agg(multi_df, "multi"), on="safe_pkg", how="outer")

    per_app["app_name"] = per_app["safe_pkg"].map(app_names).fillna(per_app["safe_pkg"])
    label = per_app["true_label_uni"].fillna(per_app["true_label_multi"])
    per_app["class"] = label.map({1: "good", 0: "bad"})
    per_app["difference"] = per_app["delta_p_img_multi"] - per_app["delta_p_img_uni"]
    per_app["app_confidence"] = per_app["safe_pkg"].map(app_confidence)

    per_app = per_app[[
        "app_name", "safe_pkg", "class", "app_confidence",
        "delta_p_img_uni", "delta_p_img_multi", "difference",
        "n_screens_uni", "n_screens_multi",
    ]]
    # Good first and, within the class, from the most to the least confident
    # app. The class order is explicit rather than alphabetical, otherwise
    # "bad" would sort ahead of "good". Sorting before rounding keeps ties on
    # the 4th decimal from swapping positions relative to the full values.
    class_rank = per_app["class"].map({"good": 0, "bad": 1})
    per_app = (
        per_app.assign(_class_rank=class_rank)
        .sort_values(["_class_rank", "app_confidence"], ascending=[True, False])
        .drop(columns="_class_rank")
        .reset_index(drop=True)
    )
    # ``difference`` is computed from the full-precision values; rounding last
    # keeps it consistent with multi minus uni.
    for col in (
        "app_confidence", "delta_p_img_uni", "delta_p_img_multi", "difference",
    ):
        per_app[col] = per_app[col].round(4)
    return per_app


def _describe_pair(
    values_uni: np.ndarray, values_multi: np.ndarray, name: str
) -> dict:
    """Descriptive stats of the paired delta_p_img (uni vs multi).

    Only pairs where both sides have a value are considered, so mean and median
    describe exactly the same set of screens on both pipelines.
    ``n_multi_maior`` counts on how many screens the multimodal delta_p_img was
    larger, which prevents drawing a conclusion from a mean that a few apps
    pull up.
    """
    valid = ~(np.isnan(values_uni) | np.isnan(values_multi))
    vu, vm = values_uni[valid], values_multi[valid]
    if not len(vu):
        return {
            "name": name, "n": 0,
            "mean_uni": float("nan"), "mean_multi": float("nan"),
            "median_uni": float("nan"), "median_multi": float("nan"),
            "n_multi_maior": 0, "winner": "—",
        }
    return {
        "name": name,
        "n": int(len(vu)),
        "mean_uni": float(np.mean(vu)),
        "mean_multi": float(np.mean(vm)),
        "median_uni": float(np.median(vu)),
        "median_multi": float(np.median(vm)),
        "n_multi_maior": int(np.sum(vm > vu)),
        "winner": "multi" if np.mean(vm) > np.mean(vu) else "uni",
    }


def _format_pair(res: dict) -> list[str]:
    """Report block with the descriptive stats of one set of paired screens."""
    n = res["n"]
    pct = f"{100.0 * res['n_multi_maior'] / n:.1f}%" if n else "—"
    return [
        f"  n (pairs):            {n}",
        f"  mean   uni:           {res['mean_uni']:.5f}",
        f"  mean   multi:         {res['mean_multi']:.5f}",
        f"  median uni:           {res['median_uni']:.5f}",
        f"  median multi:         {res['median_multi']:.5f}",
        f"  higher mean:          {res['winner']}",
        f"  screens multi > uni:  {res['n_multi_maior']}/{n} ({pct})",
    ]


def build_summary(merged: pd.DataFrame, per_app: pd.DataFrame | None = None) -> str:
    """Build comparison_summary.txt in readable form."""
    lines = []
    lines.append("=" * 78)
    lines.append("COMPARISON XAI — Unimodal × Multimodal")
    lines.append("Visual explanation faithfulness by ablation (ΔP_img)")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Paired screens: {len(merged)}")
    bom_mask = merged["uni_true_label"] == 1
    ruim_mask = merged["uni_true_label"] == 0
    lines.append(f"  - Good: {int(bom_mask.sum())}")
    lines.append(f"  - Bad:  {int(ruim_mask.sum())}")
    lines.append("")

    vu = merged["uni_delta_p_img"].to_numpy()
    vm = merged["multi_delta_p_img"].to_numpy()

    lines.append("-" * 78)
    lines.append("ΔP_img — overall (paired screens, uni vs multi)")
    lines.append("-" * 78)
    lines.extend(_format_pair(_describe_pair(vu, vm, "ΔP_img")))

    lines.append("")
    lines.append("-" * 78)
    lines.append("ΔP_img — by class")
    lines.append("-" * 78)
    for cls_name, mask in [("GOOD", bom_mask), ("BAD", ruim_mask)]:
        if mask.sum() == 0:
            continue
        lines.append(f"\n[{cls_name}] n={int(mask.sum())}")
        res = _describe_pair(
            merged.loc[mask, "uni_delta_p_img"].to_numpy(),
            merged.loc[mask, "multi_delta_p_img"].to_numpy(),
            f"ΔP_img {cls_name}",
        )
        lines.extend(_format_pair(res))

    if per_app is not None and not per_app.empty:
        lines.append("")
        lines.append("-" * 78)
        lines.append("Mean ΔP_img per app (all screens of each run)")
        lines.append("-" * 78)
        lines.append(
            f"{'App':<34s} {'class':>6s} {'conf':>7s} {'uni':>8s} "
            f"{'multi':>8s} {'diff':>8s} {'screens':>8s}"
        )
        for _, r in per_app.iterrows():
            name = str(r["app_name"])[:33]
            screens = f"{int(r['n_screens_uni'])}/{int(r['n_screens_multi'])}"
            conf = (
                f"{r['app_confidence']:.3f}"
                if pd.notna(r["app_confidence"])
                else "—"
            )
            lines.append(
                f"{name:<34s} {str(r['class']):>6s} {conf:>7s} "
                f"{r['delta_p_img_uni']:>8.3f} {r['delta_p_img_multi']:>8.3f} "
                f"{r['difference']:>8.3f} {screens:>8s}"
            )

    lines.append("")
    lines.append("=" * 78)
    lines.append("Legend:")
    lines.append("  ΔP_img = |p_orig - p_img| when the top-K most salient")
    lines.append("    regions are zeroed. Higher = more faithful explanation.")
    lines.append("  Multimodal: mean per screen over the (screen, review)")
    lines.append("    pairs. Unimodal: the single value of the screen. Same")
    lines.append("    aggregation as the faithfulness table.")
    lines.append("  higher mean: which pipeline has the larger mean")
    lines.append("  screens multi > uni: on how many screens the multimodal")
    lines.append("    ΔP_img was larger — shows whether the mean reflects the")
    lines.append("    whole set or is pulled up by a few screens.")
    lines.append("")
    lines.append("  Per-app table: each side is the mean over all screens of")
    lines.append("    the app in its own run, not only the paired ones.")
    lines.append("    Column screens = uni/multi; when the two numbers differ,")
    lines.append("    the runs did not cover the same screens of the app.")
    lines.append("    The blocks above do use only the paired screens.")
    lines.append("  conf: app confidence on the multimodal side (mean of the")
    lines.append("    screen predictions, complemented when below 0.5).")
    lines.append("    Orders the table within each class.")
    lines.append("")
    lines.append("=" * 78)

    return "\n".join(lines)


# =============================================================================
# ENTRY POINT
# =============================================================================
def compare(config: dict) -> None:
    """Entry point: read config, run the comparison, write the outputs."""
    uni_run = config.get("unimodal_run_dir")
    multi_run = config.get("multimodal_run_dir")
    if not uni_run or not multi_run:
        raise ValueError(
            "Invalid config: set 'unimodal_run_dir' and 'multimodal_run_dir'."
        )

    uni_path = Path(uni_run)
    multi_path = Path(multi_run)
    if not uni_path.exists():
        raise FileNotFoundError(f"Unimodal run not found: {uni_path}")
    if not multi_path.exists():
        raise FileNotFoundError(f"Multimodal run not found: {multi_path}")

    print(f"[compare_xai] Unimodal:   {uni_path}")
    print(f"[compare_xai] Multimodal: {multi_path}")

    uni_df = _build_per_screen_table(uni_path, "uni")
    multi_df = _build_per_screen_table(multi_path, "multi")
    merged = compare_pair(uni_df, multi_df)

    if merged.empty:
        print("[compare_xai] No screens in common — aborting.")
        return

    # Names come from the multimodal run; the unimodal one covers apps missing
    # from it.
    app_names = {**_load_app_names(uni_path), **_load_app_names(multi_path)}
    # Multimodal confidence drives the order of the per-app table.
    app_conf = _load_app_confidence(multi_path)
    per_app = build_per_app_table(uni_df, multi_df, app_names, app_conf)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_EXPLANATIONS_COMPARISON / f"comparison_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_out = out_dir / "comparison_per_screen.csv"
    merged.to_csv(merged_out, index=False)
    print(f"[compare_xai] → {merged_out} ({len(merged)} rows)")

    per_app_out = out_dir / "comparison_per_app.csv"
    # float_format pins the 4 decimals in the file; without it pandas drops
    # trailing zeros and the column width varies per row.
    per_app.to_csv(per_app_out, index=False, float_format="%.4f")
    print(f"[compare_xai] → {per_app_out} ({len(per_app)} apps)")

    summary = build_summary(merged, per_app)
    summary_path = out_dir / "comparison_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"[compare_xai] → {summary_path}")
    print()
    print(summary)

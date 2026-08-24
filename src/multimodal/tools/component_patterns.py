#!/usr/bin/env python3
"""UI component patterns per app (OBI rankings).

Offline tool: reads the CSVs of a finished explainability run and summarises,
per app, which interface components concentrate the saliency. No TF/GPU, no
re-run of the pipeline.

Builds the component-patterns table from two complementary measures, both
taken from the ``top_components`` column of
``screen_explanations.csv`` (the top-K ablated components of each screen, in
OBI ranking order):

- **Rank-0** — the component that most often comes first, reported as
  ``Component (k/n)``: in ``k`` of the app's ``n`` screens that type was the
  most salient. Shows whether the app has a stable visual focus.
- **Top-N recurring** — how many times each type appears among the top-K of
  all screens, reported as ``Component (14x)``. This is the aggregation the
  NLG module performs in ``_aggregate_top_components`` for the "most recurring
  salient components" sentence, so the CSV and the generated text agree by
  construction.

Outputs:

- ``component_patterns.csv`` — one row per app, with the table columns plus the
  individual counts for reuse.
- ``component_patterns.txt`` — the same table as readable text.

CLI entry: "UI component patterns per app (OBI rankings)".
"""

from __future__ import annotations

import datetime
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from multimodal.common.paths import RESULTS_EXPLANATIONS_COMPONENTS

# How many recurring components to report per app.
DEFAULT_TOP_N = 3


# =============================================================================
# LOADERS
# =============================================================================
def _iter_apps(run_dir: Path) -> dict[str, Path]:
    """Return {safe_pkg: app_dir} for a run laid out as ``apps/<pkg>/``."""
    apps_dir = run_dir / "apps"
    if not apps_dir.exists():
        raise FileNotFoundError(f"'apps' directory not found in {run_dir}")
    return {d.name: d for d in sorted(apps_dir.iterdir()) if d.is_dir()}


def _load_app_meta(run_dir: Path) -> dict[str, dict]:
    """Return {safe_pkg: {'app_name', 'category'}} from ``screen_explanations.json``.

    The per-app CSVs carry neither the commercial name nor the category, only
    the ``screen_id`` and the metrics. The table is identified by app name.
    """
    json_path = run_dir / "screen_explanations.json"
    if not json_path.exists():
        return {}
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[component_patterns] Failed to read {json_path}: {e}")
        return {}
    out: dict[str, dict] = {}
    for scr in data:
        pkg = scr.get("package_name")
        if not pkg:
            continue
        safe = str(pkg).replace(".", "_")
        if safe not in out:
            out[safe] = {
                "package_name": str(pkg),
                "app_name": scr.get("app_name") or str(pkg),
                "category": scr.get("category") or "",
            }
    return out


def _load_app_classes(run_dir: Path) -> dict[str, str]:
    """Return {safe_pkg: 'good'|'bad'} from ``app_explanations.json``.

    Uses ``app_prediction``, the class the model predicted (mean of the screen
    predictions), not the true class: the component patterns describe what the
    model leaned on to decide, so the class that contextualises the row is the
    one it assigned.
    """
    json_path = run_dir / "app_explanations.json"
    if not json_path.exists():
        return {}
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[component_patterns] Failed to read {json_path}: {e}")
        return {}
    out: dict[str, str] = {}
    for app in data:
        pkg = app.get("package_name")
        cls = app.get("app_prediction")
        if pkg and cls:
            out[str(pkg).replace(".", "_")] = str(cls)
    return out


def _screen_components(app_dir: Path) -> list[list[str]]:
    """List of lists: the top-K components of each screen of the app, in order.

    Screens without ``top_components`` are skipped: with no OBI ranking there
    is no pattern to extract from them.
    """
    sc_path = app_dir / "screen_explanations.csv"
    if not sc_path.exists():
        return []
    try:
        sc = pd.read_csv(sc_path)
    except Exception as e:
        print(f"[component_patterns] Failed to load {sc_path}: {e}")
        return []
    if "top_components" not in sc.columns:
        return []
    screens: list[list[str]] = []
    for raw in sc["top_components"]:
        if pd.isna(raw):
            continue
        comps = [c.strip() for c in str(raw).split(",") if c.strip()]
        if comps:
            screens.append(comps)
    return screens


# =============================================================================
# AGGREGATION
# =============================================================================
def app_pattern(screens: list[list[str]], top_n: int = DEFAULT_TOP_N) -> dict:
    """Summarise one app's component patterns from the per-screen top-K."""
    n_screens = len(screens)
    rank0 = Counter(comps[0] for comps in screens if comps)
    recurring = Counter(c for comps in screens for c in comps)

    rank0_comp, rank0_hits = ("", 0)
    if rank0:
        rank0_comp, rank0_hits = rank0.most_common(1)[0]

    top = recurring.most_common(top_n)
    return {
        "n_screens": n_screens,
        "rank0_component": rank0_comp,
        "rank0_screens": rank0_hits,
        "rank0_fmt": f"{rank0_comp} ({rank0_hits}/{n_screens})" if rank0_comp else "",
        "top_components_fmt": ", ".join(f"{c} ({k}x)" for c, k in top),
        "_top": top,
    }


def build_table(
    run_dir: Path, packages: list[str] | None, top_n: int = DEFAULT_TOP_N
) -> pd.DataFrame:
    """Build the component-patterns table, one row per app.

    Args:
        run_dir: explainability run laid out as ``apps/<pkg>/``.
        packages: ``package_name`` values to include, in the order they should
            appear. When None or empty, processes every app of the run in
            alphabetical order.
        top_n: how many recurring components to report.
    """
    apps = _iter_apps(run_dir)
    meta = _load_app_meta(run_dir)
    classes = _load_app_classes(run_dir)

    if packages:
        wanted = [(p, p.replace(".", "_")) for p in packages]
        missing = [p for p, safe in wanted if safe not in apps]
        if missing:
            print(
                f"[component_patterns] WARNING: {len(missing)} app(s) from the "
                f"config are missing in the run: {', '.join(missing)}"
            )
        selected = [(p, safe) for p, safe in wanted if safe in apps]
    else:
        selected = [
            (meta.get(safe, {}).get("package_name", safe), safe)
            for safe in apps
        ]

    rows = []
    for pkg, safe in selected:
        screens = _screen_components(apps[safe])
        if not screens:
            print(f"[component_patterns] {pkg}: no top_components — skipping.")
            continue
        pat = app_pattern(screens, top_n=top_n)
        info = meta.get(safe, {})
        row = {
            "app_name": info.get("app_name", pkg),
            "package_name": info.get("package_name", pkg),
            "class": classes.get(safe, ""),
            "category": info.get("category", ""),
            "n_screens": pat["n_screens"],
            "rank0": pat["rank0_fmt"],
            "top_components": pat["top_components_fmt"],
            "rank0_component": pat["rank0_component"],
            "rank0_screens": pat["rank0_screens"],
        }
        # One column per position, so the counts can be reused without parsing
        # the formatted string.
        for i in range(top_n):
            comp, cnt = pat["_top"][i] if i < len(pat["_top"]) else ("", 0)
            row[f"top{i + 1}_component"] = comp
            row[f"top{i + 1}_n"] = cnt
        rows.append(row)

    return pd.DataFrame(rows)


def build_report(table: pd.DataFrame, run_dir: Path) -> str:
    """Build component_patterns.txt in readable form."""
    lines = []
    lines.append("=" * 78)
    lines.append("UI COMPONENT PATTERNS PER APP (OBI rankings)")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Run: {run_dir}")
    lines.append(f"Apps: {len(table)}")
    if not table.empty:
        lines.append(f"Screens total: {int(table['n_screens'].sum())}")
    lines.append("")
    lines.append("-" * 78)
    lines.append(
        f"{'App':<32s} {'class':>6s} {'scr':>5s} {'rank-0':>14s}  "
        f"{'top recurring components':<28s}"
    )
    lines.append("-" * 78)
    for _, r in table.iterrows():
        lines.append(
            f"{str(r['app_name'])[:31]:<32s} {str(r['class']):>6s} "
            f"{int(r['n_screens']):>5d} {str(r['rank0']):>14s}  "
            f"{str(r['top_components'])}"
        )
    lines.append("")
    lines.append("=" * 78)
    lines.append("Legend:")
    lines.append("  rank-0 = most salient component of the screen; (k/n) = in k")
    lines.append("    of the app's n screens that type came first. A low k means")
    lines.append("    the app has no stable visual focus.")
    lines.append("  top components = how many times each type appears among the")
    lines.append("    top-K of all screens of the app. Same aggregation the NLG")
    lines.append("    uses for 'most recurring salient components'.")
    lines.append("  Screens with no OBI ranking (empty top_components column)")
    lines.append("    are left out of the counts.")
    lines.append("")
    lines.append("Reference:")
    lines.append("  Leiva et al. 2020 (UI component rankings)")
    lines.append("=" * 78)
    return "\n".join(lines)


# =============================================================================
# ENTRY POINT
# =============================================================================
def generate(config: dict) -> None:
    """Entry point: read config, build the table, write the outputs."""
    run = config.get("run_dir")
    if not run:
        raise ValueError("Invalid config: set 'run_dir'.")

    run_path = Path(run)
    if not run_path.exists():
        raise FileNotFoundError(f"Run not found: {run_path}")

    packages = config.get("packages") or []
    top_n = int(config.get("top_n", DEFAULT_TOP_N))

    print(f"[component_patterns] Run: {run_path}")
    print(
        f"[component_patterns] Apps: "
        f"{'all in the run' if not packages else f'{len(packages)} from config'}"
        f" | top-N: {top_n}"
    )

    table = build_table(run_path, packages, top_n=top_n)
    if table.empty:
        print("[component_patterns] No app with an OBI ranking — aborting.")
        return

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_EXPLANATIONS_COMPONENTS / f"components_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_out = out_dir / "component_patterns.csv"
    table.to_csv(csv_out, index=False)
    print(f"[component_patterns] → {csv_out} ({len(table)} apps)")

    report = build_report(table, run_path)
    txt_out = out_dir / "component_patterns.txt"
    txt_out.write_text(report, encoding="utf-8")
    print(f"[component_patterns] → {txt_out}")
    print()
    print(report)

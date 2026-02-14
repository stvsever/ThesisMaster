#!/usr/bin/env python3
"""
generate_pipeline_research_report.py

Create communication-ready summaries for a single integrated pipeline run.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{ts()}] {message}", flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_read_csv(path: Path) -> pd.DataFrame:
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] > 1 or sep == ",":
                return df
        except Exception:
            pass
    return pd.read_csv(path, engine="python")


def _save_figure_multi(fig: plt.Figure, png_path: Path, *, metadata: Dict[str, Any]) -> List[str]:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path = png_path.with_suffix(".svg")
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    payload = dict(metadata)
    payload["generated_at"] = datetime.now().isoformat(timespec="seconds")
    payload["files"] = [str(png_path), str(svg_path), str(pdf_path)]
    png_path.with_suffix(".figure.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return [str(png_path), str(svg_path), str(pdf_path), str(png_path.with_suffix(".figure.json"))]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_run_root = repo_root / "evaluation/05_integrated_pipeline_runs"

    parser = argparse.ArgumentParser(
        description="Generate communication-ready report files for one integrated pipeline run."
    )
    parser.add_argument(
        "--run-root",
        type=str,
        required=True,
        help="Path to one integrated run directory (contains pipeline_summary.json).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Optional report output path. Defaults to <run-root>/05_research_reports.",
    )
    parser.add_argument("--max-profiles", type=int, default=0)
    parser.add_argument("--include-empty-profiles", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--default-run-parent",
        type=str,
        default=str(default_run_root),
        help="For provenance only; no processing dependency.",
    )
    return parser.parse_args()


def _read_readiness(profile_root: Path) -> Dict[str, Any]:
    readiness_path = profile_root / "readiness_report.json"
    if not readiness_path.exists():
        return {
            "has_readiness": False,
            "readiness_label": None,
            "readiness_score": None,
            "recommended_tier": None,
            "tier3_variant": None,
            "analysis_set": None,
        }
    payload = read_json(readiness_path)
    overall = payload.get("overall", {}) or {}
    plan = overall.get("analysis_execution_plan", {}) or {}
    return {
        "has_readiness": True,
        "readiness_label": overall.get("readiness_label"),
        "readiness_score": overall.get("readiness_score_0_100"),
        "recommended_tier": overall.get("recommended_tier"),
        "tier3_variant": overall.get("tier3_variant"),
        "analysis_set": plan.get("analysis_set"),
    }


def _read_network(profile_root: Path) -> Dict[str, Any]:
    cmp_path = profile_root / "comparison_summary.json"
    if not cmp_path.exists():
        return {
            "has_network": False,
            "network_analysis_set": None,
            "network_methods": None,
        }
    payload = read_json(cmp_path)
    execution_plan = payload.get("execution_plan", {}) or {}
    method_status = payload.get("method_status", {}) or {}
    return {
        "has_network": True,
        "network_analysis_set": execution_plan.get("analysis_set"),
        "network_methods": method_status,
    }


def _read_impact(profile_root: Path) -> Dict[str, Any]:
    composite_path = profile_root / "predictor_composite.csv"
    momentary_path = profile_root / "momentary_impact.json"
    if not composite_path.exists():
        return {
            "has_impact": False,
            "impact_source": None,
            "top_predictor": None,
            "top_predictor_impact": None,
            "n_predictors_ranked": 0,
            "has_visuals": False,
            "n_visual_files": 0,
        }
    df = safe_read_csv(composite_path)
    if not df.empty and "predictor_impact" in df.columns:
        top_row = df.sort_values("predictor_impact", ascending=False).iloc[0]
        top_predictor = str(top_row.get("predictor", ""))
        top_impact = float(top_row.get("predictor_impact", 0.0))
    else:
        top_predictor = None
        top_impact = None

    impact_source: Optional[str] = None
    if momentary_path.exists():
        payload = read_json(momentary_path)
        impact_source = ((payload.get("meta", {}) or {}).get("impact_source"))

    visuals_dir = profile_root / "visuals"
    n_visual_files = len(list(visuals_dir.glob("*.png"))) if visuals_dir.exists() else 0
    return {
        "has_impact": True,
        "impact_source": impact_source,
        "top_predictor": top_predictor,
        "top_predictor_impact": top_impact,
        "n_predictors_ranked": int(len(df)),
        "has_visuals": visuals_dir.exists(),
        "n_visual_files": int(n_visual_files),
    }


def _read_handoff(profile_root: Path) -> Dict[str, Any]:
    csv_path = profile_root / "top_treatment_target_candidates.csv"
    trace_path = profile_root / "step03_prompt_trace.json"
    if not csv_path.exists():
        return {"has_handoff": False, "handoff_candidates": 0, "handoff_mode": None}
    df = safe_read_csv(csv_path)
    mode = None
    if trace_path.exists():
        trace = read_json(trace_path)
        mode = trace.get("reason")
    return {"has_handoff": True, "handoff_candidates": int(len(df)), "handoff_mode": mode}


def _read_intervention(profile_root: Path) -> Dict[str, Any]:
    json_path = profile_root / "step05_hapa_intervention.json"
    trace_path = profile_root / "step05_hapa_prompt_trace.json"
    if not json_path.exists():
        return {"has_intervention": False, "intervention_targets": 0, "intervention_barriers": 0, "intervention_mode": None}
    payload = read_json(json_path)
    mode = None
    if trace_path.exists():
        trace = read_json(trace_path)
        mode = trace.get("reason")
    return {
        "has_intervention": True,
        "intervention_targets": int(len(payload.get("selected_treatment_targets", []) or [])),
        "intervention_barriers": int(len(payload.get("selected_barriers", []) or [])),
        "intervention_coping": int(len(payload.get("selected_coping_strategies", []) or [])),
        "intervention_mode": mode,
    }


def _collect_profiles(
    run_root: Path,
    summary: Dict[str, Any],
    max_profiles: int,
    include_empty_profiles: bool,
) -> List[Dict[str, Any]]:
    readiness_root = run_root / "00_readiness_check"
    network_root = run_root / "01_time_series_analysis/network"
    impact_root = run_root / "02_momentary_impact_coefficients"
    handoff_root = run_root / "03_treatment_target_handoff"
    intervention_root = run_root / "03b_translation_digital_intervention"

    summary_profiles = [str(p) for p in (summary.get("profiles") or [])]
    discovered_profiles = set(summary_profiles)

    for root in [readiness_root, network_root, impact_root, handoff_root, intervention_root]:
        if root.exists():
            for child in root.iterdir():
                if child.is_dir():
                    discovered_profiles.add(child.name)

    ordered = sorted(discovered_profiles)
    if max_profiles > 0:
        ordered = ordered[:max_profiles]

    rows: List[Dict[str, Any]] = []
    for profile_id in ordered:
        row: Dict[str, Any] = {"profile_id": profile_id}
        row.update(_read_readiness(readiness_root / profile_id))
        row.update(_read_network(network_root / profile_id))
        row.update(_read_impact(impact_root / profile_id))
        row.update(_read_handoff(handoff_root / profile_id))
        row.update(_read_intervention(intervention_root / profile_id))
        if include_empty_profiles:
            rows.append(row)
        else:
            if any(
                [
                    bool(row.get("has_readiness")),
                    bool(row.get("has_network")),
                    bool(row.get("has_impact")),
                    bool(row.get("has_handoff")),
                    bool(row.get("has_intervention")),
                ]
            ):
                rows.append(row)
    return rows


def _iterative_memory_summary(summary: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
    iterative = summary.get("iterative_memory", {}) or {}
    history_root = iterative.get("history_root")
    run_id = str(summary.get("run_id") or "")
    out: Dict[str, Any] = {
        "enabled": bool(iterative.get("enabled", False)),
        "history_root": str(history_root or ""),
        "lineage_rows": 0,
        "snapshot_rows": 0,
        "plot_files": [],
    }
    if not history_root:
        return out
    history_path = Path(str(history_root)).expanduser().resolve()
    snapshots_path = history_path / "feature_snapshots.parquet"
    lineage_path = history_path / "model_lineage.parquet"
    if lineage_path.exists():
        lineage = pd.read_parquet(lineage_path)
        if "run_id" in lineage.columns and run_id:
            lineage = lineage.loc[lineage["run_id"].astype(str) == run_id].copy()
        out["lineage_rows"] = int(len(lineage))
    if snapshots_path.exists():
        snapshots = pd.read_parquet(snapshots_path)
        if "run_id" in snapshots.columns and run_id:
            snapshots = snapshots.loc[snapshots["run_id"].astype(str) == run_id].copy()
        out["snapshot_rows"] = int(len(snapshots))
        if not snapshots.empty and "cycle_index" in snapshots.columns:
            grp = snapshots.groupby("cycle_index", as_index=False).size()
            if len(grp) > 1:
                fig = plt.figure(figsize=(7.5, 4.2))
                ax = fig.add_subplot(111)
                ax.plot(grp["cycle_index"], grp["size"], marker="o", color="#1d3557")
                ax.set_xlabel("Cycle Index")
                ax.set_ylabel("Profiles Logged")
                ax.set_title("PHOENIX Iterative Memory Progression")
                ax.grid(alpha=0.25)
                out["plot_files"] = _save_figure_multi(
                    fig,
                    output_root / "cross_cycle_progression.png",
                    metadata={
                        "plot_type": "cross_cycle_progression",
                        "history_root": str(history_path),
                    },
                )
    return out


def _build_markdown(
    summary: Dict[str, Any],
    profile_rows: List[Dict[str, Any]],
    output_root: Path,
    iterative_summary: Dict[str, Any],
) -> str:
    run_id = str(summary.get("run_id", "unknown"))
    status = str(summary.get("status", "unknown"))
    stage_results = summary.get("stage_results") or []
    outputs = summary.get("outputs") or {}

    labels = [str(row.get("readiness_label")) for row in profile_rows if row.get("readiness_label")]
    label_counts = Counter(labels)
    methods_counter = Counter()
    handoff_modes = Counter()
    intervention_modes = Counter()
    for row in profile_rows:
        methods = row.get("network_methods") or {}
        for method_name, method_status in methods.items():
            methods_counter[f"{method_name}:{method_status}"] += 1
        if row.get("handoff_mode"):
            handoff_modes[str(row.get("handoff_mode"))] += 1
        if row.get("intervention_mode"):
            intervention_modes[str(row.get("intervention_mode"))] += 1

    lines: List[str] = []
    lines.append(f"# PHOENIX Engine — Integrated Run Report (`{run_id}`)")
    lines.append("")
    lines.append("## Run Summary")
    lines.append(f"- Status: `{status}`")
    lines.append(f"- Profiles in report: `{len(profile_rows)}`")
    lines.append(f"- Stage results logged: `{len(stage_results)}`")
    lines.append(f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("## Component Coverage")
    lines.append(f"- Readiness available: `{sum(1 for r in profile_rows if r.get('has_readiness'))}`")
    lines.append(f"- Network available: `{sum(1 for r in profile_rows if r.get('has_network'))}`")
    lines.append(f"- Impact available: `{sum(1 for r in profile_rows if r.get('has_impact'))}`")
    lines.append(f"- Handoff available: `{sum(1 for r in profile_rows if r.get('has_handoff'))}`")
    lines.append(f"- Step-05 intervention available: `{sum(1 for r in profile_rows if r.get('has_intervention'))}`")
    lines.append(f"- Visualizations available: `{sum(1 for r in profile_rows if r.get('has_visuals'))}`")
    lines.append("")
    lines.append("## Readiness Label Distribution")
    if label_counts:
        for label, count in sorted(label_counts.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- `{label}`: `{count}`")
    else:
        lines.append("- No readiness labels found.")
    lines.append("")
    lines.append("## Network Method Execution")
    if methods_counter:
        for key, count in sorted(methods_counter.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- `{key}`: `{count}`")
    else:
        lines.append("- No method execution metadata found.")
    lines.append("")
    lines.append("## LLM Runtime Modes")
    if handoff_modes:
        lines.append("- Step-03/04 modes:")
        for key, count in sorted(handoff_modes.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"  - `{key}`: `{count}`")
    else:
        lines.append("- No Step-03/04 mode traces found.")
    if intervention_modes:
        lines.append("- Step-05 modes:")
        for key, count in sorted(intervention_modes.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"  - `{key}`: `{count}`")
    else:
        lines.append("- No Step-05 mode traces found.")
    lines.append("")
    lines.append("## Iterative Memory")
    lines.append(f"- Enabled: `{iterative_summary.get('enabled')}`")
    lines.append(f"- Snapshot rows: `{iterative_summary.get('snapshot_rows')}`")
    lines.append(f"- Lineage rows: `{iterative_summary.get('lineage_rows')}`")
    if iterative_summary.get("plot_files"):
        lines.append(f"- Cross-cycle plot files: `{len(iterative_summary.get('plot_files', []))}`")
    lines.append("")
    lines.append("## Output Paths")
    for key in sorted(outputs.keys()):
        lines.append(f"- `{key}`: `{outputs[key]}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Current run scope is synthetic-data evaluation.")
    lines.append("- This report structure is forward-compatible with future real-world frontend/backend data flows.")
    lines.append(f"- Report directory: `{output_root}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if str(args.output_root).strip()
        else (run_root / "05_research_reports").resolve()
    )
    ensure_dir(output_root)
    summary_path = run_root / "pipeline_summary.json"

    if not run_root.exists():
        log(f"[ERROR] run-root not found: {run_root}")
        return 2
    if not summary_path.exists():
        log(f"[ERROR] pipeline summary not found: {summary_path}")
        return 3

    summary = read_json(summary_path)
    profile_rows = _collect_profiles(
        run_root=run_root,
        summary=summary,
        max_profiles=int(args.max_profiles),
        include_empty_profiles=bool(args.include_empty_profiles),
    )

    overview_df = pd.DataFrame(profile_rows)
    if not overview_df.empty:
        overview_df = overview_df.sort_values("profile_id").reset_index(drop=True)

    component_status = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_root": str(run_root),
        "run_id": summary.get("run_id"),
        "status": summary.get("status"),
        "n_profiles_reported": int(len(profile_rows)),
        "n_with_readiness": int(sum(1 for r in profile_rows if r.get("has_readiness"))),
        "n_with_network": int(sum(1 for r in profile_rows if r.get("has_network"))),
        "n_with_impact": int(sum(1 for r in profile_rows if r.get("has_impact"))),
        "n_with_handoff": int(sum(1 for r in profile_rows if r.get("has_handoff"))),
        "n_with_intervention": int(sum(1 for r in profile_rows if r.get("has_intervention"))),
        "n_with_visualizations": int(sum(1 for r in profile_rows if r.get("has_visuals"))),
        "n_handoff_structured_llm": int(sum(1 for r in profile_rows if str(r.get("handoff_mode") or "") == "structured_llm_success")),
        "n_handoff_fallback": int(sum(1 for r in profile_rows if "fallback" in str(r.get("handoff_mode") or ""))),
        "n_intervention_structured_llm": int(sum(1 for r in profile_rows if str(r.get("intervention_mode") or "") == "structured_llm_success")),
        "n_intervention_fallback": int(sum(1 for r in profile_rows if "fallback" in str(r.get("intervention_mode") or ""))),
        "default_run_parent": str(args.default_run_parent),
    }
    iterative_summary = _iterative_memory_summary(summary=summary, output_root=output_root)
    component_status["iterative_memory"] = iterative_summary

    md = _build_markdown(
        summary=summary,
        profile_rows=profile_rows,
        output_root=output_root,
        iterative_summary=iterative_summary,
    )
    report_json = {
        "component_status": component_status,
        "pipeline_summary": summary,
        "iterative_memory": iterative_summary,
        "profiles": profile_rows,
    }

    (output_root / "run_report.md").write_text(md, encoding="utf-8")
    (output_root / "run_report.json").write_text(json.dumps(report_json, indent=2), encoding="utf-8")
    pd.DataFrame([component_status]).to_csv(output_root / "component_status.csv", index=False)
    if not overview_df.empty:
        overview_df.to_csv(output_root / "profile_overview.csv", index=False)
    else:
        pd.DataFrame(columns=["profile_id"]).to_csv(output_root / "profile_overview.csv", index=False)

    log(f"[DONE] report generated at {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

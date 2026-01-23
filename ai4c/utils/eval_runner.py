import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

from ai4c.utils.multi_round_utils import (
    sh_quote,
    parse_float,
    tail_lines,
    parse_rectified_speedup,
    read_text,
    truncate_text,
)


@dataclass
class EvalResult:
    task_path: str
    entry_sh: str
    exit_code: int
    combined_output: str
    validation_log_path: Optional[str]
    aggregated_score_path: Optional[str]
    status: str  # "success" | "failed" | "unknown"
    speedup_e2e: Optional[float]
    speedup_gpu: Optional[float]
    rectified_speedup: Optional[float]
    error_excerpt: Optional[str]


_RE_SPEEDUP_E2E = re.compile(r"\[Speedup\]\[e2e\]:\s*([0-9.]+)")
_RE_SPEEDUP_GPU = re.compile(r"\[Speedup\]\[gpu\]:\s*([0-9.]+)")
_RE_STATUS = re.compile(r"\[Result\]\s+status:\s*(success|failed)")


def format_eval_feedback(ev: Any) -> str:
    parts: list[str] = []
    parts.append(f"entry_sh={getattr(ev, 'entry_sh', None)}")
    parts.append(f"exit_code={getattr(ev, 'exit_code', None)}")
    parts.append(f"validation_log_path={getattr(ev, 'validation_log_path', None)}")
    parts.append(f"aggregated_score_path={getattr(ev, 'aggregated_score_path', None)}")
    parts.append(f"status={getattr(ev, 'status', None)}")

    speedup_e2e = getattr(ev, "speedup_e2e", None)
    if speedup_e2e is not None:
        parts.append(f"speedup_e2e={speedup_e2e}")
    speedup_gpu = getattr(ev, "speedup_gpu", None)
    if speedup_gpu is not None:
        parts.append(f"speedup_gpu={speedup_gpu}")
    rectified_speedup = getattr(ev, "rectified_speedup", None)
    if rectified_speedup is not None:
        parts.append(f"rectified_speedup={rectified_speedup}")

    error_excerpt = getattr(ev, "error_excerpt", None)
    if error_excerpt:
        parts.append("\n--- validation.log excerpt ---\n" + error_excerpt)
    else:
        parts.append("\n--- entry.sh output (tail) ---\n" + (getattr(ev, "combined_output", "") or ""))
    return "\n".join(parts)


def format_last_pass_artifacts(task_path: str, last_engineer_plan_json: str) -> str:
    if not last_engineer_plan_json:
        return ""
    try:
        plan = json.loads(last_engineer_plan_json)
    except Exception:
        return truncate_text(last_engineer_plan_json, 20000)

    pass_details = plan.get("pass_details", []) if isinstance(plan, dict) else []
    names: list[str] = []
    for p in pass_details:
        if isinstance(p, dict) and p.get("name"):
            names.append(p["name"])

    lines: list[str] = []
    lines.append("Last round pass_order: " + str(plan.get("pass_order") if isinstance(plan, dict) else names))
    lines.append("Last round pass_names: " + ", ".join(names))

    base_dir = os.path.join(task_path, "pass_dir")
    for name in names:
        fp = os.path.join(base_dir, f"{name}.py")
        try:
            code = read_text(fp)
        except Exception as e:
            code = f"<failed to read {fp}: {type(e).__name__}: {e}>"
        lines.append(f"\n--- pass_file: {fp} ---\n{truncate_text(code, 12000)}")

    # After we have captured the artifacts for prompting, remove pass .py files so
    # the next round starts from a clean slate (Engineer will rewrite them).
    try:
        if os.path.isdir(base_dir):
            for fn in os.listdir(base_dir):
                if fn.endswith(".py"):
                    try:
                        os.remove(os.path.join(base_dir, fn))
                    except Exception:
                        pass
    except Exception:
        pass

    return truncate_text("\n".join(lines), 30000)


def on_before_round(ctx: Any) -> None:
    if getattr(ctx, "last_eval", None) is None:
        return
    last_eval = ctx.last_eval
    feedback = format_eval_feedback(last_eval)
    ctx.initial_message.meta_info["last_run_feedback"] = feedback
    ctx.initial_message.meta_info["last_run_status"] = getattr(last_eval, "status", None)
    ctx.initial_message.meta_info["last_run_rectified_speedup"] = getattr(
        last_eval, "rectified_speedup", None
    )
    ctx.initial_message.meta_info["last_run_speedup_gpu"] = getattr(last_eval, "speedup_gpu", None)
    os.environ["AI4C_LAST_RUN_FEEDBACK"] = feedback

    last_plan_json = getattr(getattr(ctx, "final_message", None), "code_content", None)
    if last_plan_json:
        artifacts = format_last_pass_artifacts(ctx.initial_message.meta_info["task_path"], last_plan_json)
        ctx.initial_message.meta_info["last_pass_artifacts"] = artifacts
        os.environ["AI4C_LAST_PASS_ARTIFACTS"] = artifacts

        try:
            last_plan = json.loads(last_plan_json)
            last_pass_order = last_plan.get("pass_order") if isinstance(last_plan, dict) else None
        except Exception:
            last_pass_order = None
        if isinstance(last_pass_order, list) and last_pass_order:
            os.environ["AI4C_FIXED_PASS_NAMES"] = json.dumps(last_pass_order, ensure_ascii=False)
            ctx.initial_message.meta_info["fixed_pass_names"] = last_pass_order


def on_after_round(ctx: Any, *, eval_output_dir: Optional[str]) -> EvalResult:
    print(f"[AI4C] ===== Round {ctx.turn_idx+1}/{ctx.max_turns}: running entry.sh =====")
    ev = run_entry_and_collect(
        ctx.initial_message.meta_info["task_path"],
        eval_output_dir=eval_output_dir,
    )
    print(
        f"[AI4C] ===== Round {ctx.turn_idx+1}/{ctx.max_turns}: eval status={ev.status} rectified_speedup={ev.rectified_speedup} ====="
    )
    return ev


def run_entry_and_collect(
    task_path: str,
    *,
    eval_output_dir: Optional[str] = None,
    timeout_s: int = 1800,
) -> EvalResult:
    task_dir = Path(task_path).resolve()
    entry_sh = str((task_dir / "entry.sh").resolve())

    proc = subprocess.run(
        ["bash", "-lc", f"cd {sh_quote(str(task_dir))} && ./entry.sh"],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=os.environ.copy(),
    )
    combined = (proc.stdout or "") + ("\n" if proc.stdout and proc.stderr else "") + (proc.stderr or "")

    if eval_output_dir:
        expanded = os.path.expandvars(os.path.expanduser(eval_output_dir))
        out_path = Path(expanded).resolve()
        if out_path.is_file() and out_path.name == "validation.log":
            validation_log = out_path
            aggregated_score = out_path.parent / "aggregated_score.json"
        else:
            validation_log = out_path / "validation.log"
            aggregated_score = out_path / "aggregated_score.json"
    else:
        out_path = task_dir / "workspace_graph_net_bench_test"
        validation_log = out_path / "validation.log"
        aggregated_score = out_path / "aggregated_score.json"

    if not validation_log.exists():
        raise FileNotFoundError(
            "validation.log not found.\n"
            f"- expected: {validation_log}\n"
            f"- hint: ensure entry.sh writes logs into eval_output_dir or {out_path}\n"
            f"- entry.sh exit_code={proc.returncode}\n"
            f"- entry.sh output (tail):\n{tail_lines(combined, max_lines=120)}"
        )
    if not aggregated_score.exists():
        raise FileNotFoundError(
            "aggregated_score.json not found.\n"
            f"- expected: {aggregated_score}\n"
            f"- hint: ensure entry.sh writes scores into eval_output_dir or {out_path}\n"
            f"- entry.sh exit_code={proc.returncode}\n"
            f"- validation.log (tail):\n{tail_lines(validation_log.read_text(errors='replace'), max_lines=120)}"
        )

    validation_log_path = str(validation_log)
    aggregated_score_path = str(aggregated_score)

    log_text = validation_log.read_text(errors="replace")
    status_m = _RE_STATUS.search(log_text)
    status = status_m.group(1) if status_m else "unknown"

    m_e2e = _RE_SPEEDUP_E2E.search(log_text)
    speedup_e2e = parse_float(m_e2e.group(1)) if m_e2e else None
    m_gpu = _RE_SPEEDUP_GPU.search(log_text)
    speedup_gpu = parse_float(m_gpu.group(1)) if m_gpu else None
    rectified = parse_rectified_speedup(aggregated_score_path)

    error_excerpt = None
    if status == "failed" or "Traceback" in log_text or "CompilationError" in log_text:
        error_excerpt = tail_lines(log_text, max_lines=120)

    return EvalResult(
        task_path=str(task_dir),
        entry_sh=entry_sh,
        exit_code=proc.returncode,
        combined_output=tail_lines(combined, max_lines=200),
        validation_log_path=validation_log_path,
        aggregated_score_path=aggregated_score_path,
        status=status,
        speedup_e2e=speedup_e2e,
        speedup_gpu=speedup_gpu,
        rectified_speedup=rectified,
        error_excerpt=error_excerpt,
    )



import json
import math
import os
import re
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd

from src.client import TransformersClient, LLMConfig
from src.runner import run_llm_dataframe, RunConfig
from src.run3_prompts import SYSTEM_PROMPT, build_user_prompt
from src.run3_config import build_mask, OUTPUT_COLS

# ---------------------------------------------------------------------------
# Valid stance values
# ---------------------------------------------------------------------------
_VALID_STANCES = {"CRITICIZED", "PRAISED", "NEUTRAL"}


def parse_output(raw: str) -> dict:
    """
    Parse LLM response into {"keyword_stances": '{"KW": "CRITICIZED", ...}'}.

    Expected format (one line per keyword):
        BAG: CRITICIZED
        OFSP: NEUTRAL
        SECO: PRAISED
    """
    empty = {col: pd.NA for col in OUTPUT_COLS}
    if not raw:
        return empty

    stances: dict[str, str] = {}
    for line in str(raw).strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Remove optional leading bullet/dash
        line = re.sub(r"^[-•*]\s*", "", line)
        colon_idx = line.find(":")
        if colon_idx == -1:
            continue
        kw = line[:colon_idx].strip()
        stance = line[colon_idx + 1:].strip().upper()
        # Accept truncated variants too (e.g. "CRITICIZ" → "CRITICIZED")
        for valid in _VALID_STANCES:
            if valid.startswith(stance) or stance.startswith(valid[:5]):
                stance = valid
                break
        if kw and stance in _VALID_STANCES:
            stances[kw] = stance

    if not stances:
        return empty

    return {"keyword_stances": json.dumps(stances, ensure_ascii=False)}


def main() -> int:
    ap = argparse.ArgumentParser()

    # --- I/O ---
    ap.add_argument("--input",        required=True)
    ap.add_argument("--output_base",  required=True)
    ap.add_argument("--text_col",     default="text")
    ap.add_argument("--n_rows",       type=int, default=0)

    # --- Model ---
    ap.add_argument("--model_path",        required=True)
    ap.add_argument("--dtype",             required=True, choices=["bf16", "fp16", "auto"])
    ap.add_argument("--backend",           default="transformers", choices=["vllm", "transformers"])
    ap.add_argument("--trust_remote_code", action="store_true")

    # --- Inference ---
    ap.add_argument("--batch_size",        required=True, type=int)
    ap.add_argument("--temperature",       required=True, type=float)
    ap.add_argument("--max_new_tokens",    required=True, type=int)
    ap.add_argument("--max_input_tokens",  required=True, type=int)

    # --- Internal ---
    ap.add_argument("--job_id",    default=None)
    ap.add_argument("--task_id",   type=int, default=None,
                    help="Array task index (0-indexed). If omitted, read from SLURM_ARRAY_TASK_ID.")
    ap.add_argument("--num_tasks", type=int, default=None,
                    help="Total number of array tasks. If omitted, read from SLURM_ARRAY_TASK_COUNT.")

    args = ap.parse_args()

    task_id = args.task_id
    if task_id is None and os.environ.get("SLURM_ARRAY_TASK_ID"):
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    num_tasks = args.num_tasks
    if num_tasks is None and os.environ.get("SLURM_ARRAY_TASK_COUNT"):
        num_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])

    # Checkpoint is task-scoped so it survives re-submissions
    checkpoint_path = None
    if task_id is not None:
        checkpoint_path = args.output_base + f"_task{task_id}_checkpoint.parquet"

    # Load or resume from checkpoint
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"[resume] Loading checkpoint: {checkpoint_path}", flush=True)
        df = pd.read_parquet(checkpoint_path)
    else:
        df = (pd.read_parquet(args.input)
              if args.input.endswith(".parquet")
              else pd.read_csv(args.input, low_memory=False))

        if args.n_rows > 0:
            df = df.head(args.n_rows).copy()
            print(f"[n_rows] Subsetting to first {args.n_rows} rows")

        if task_id is not None and num_tasks is not None:
            chunk_size = math.ceil(len(df) / num_tasks)
            start = task_id * chunk_size
            end   = min(start + chunk_size, len(df))
            print(
                f"[pipeline] Array task {task_id}/{num_tasks} — "
                f"rows {start}:{end} ({end - start} rows)",
                flush=True,
            )
            df = df.iloc[start:end].copy()

    for col in OUTPUT_COLS:
        if col not in df.columns:
            df[col] = pd.Series(pd.NA, index=df.index, dtype="string")

    client = TransformersClient(
        LLMConfig(
            model_path=args.model_path,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            backend=args.backend,
        )
    )

    run_cfg = RunConfig(
        id_col="article_id" if "article_id" in df.columns else "__index__",
        text_col=args.text_col,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
    )

    mask = build_mask(df, text_col=args.text_col)

    out = run_llm_dataframe(
        df=df,
        cfg=run_cfg,
        client=client,
        system_prompt=SYSTEM_PROMPT,
        select_mask_fn=lambda df_: mask,
        build_prompt_fn=lambda row, col: build_user_prompt(row, col),
        parse_fn=parse_output,
        output_cols=OUTPUT_COLS,
        skip_if_already_filled=OUTPUT_COLS[0],   # resume on keyword_stances
        checkpoint_path=checkpoint_path,
        checkpoint_every=50,
    )

    job_id = (
        os.environ.get("SLURM_ARRAY_JOB_ID")
        or os.environ.get("SLURM_JOB_ID")
        or args.job_id
        or "nojobid"
    )
    if task_id is not None:
        base = f"{args.output_base}_task{task_id}_job{job_id}"
    else:
        base = f"{args.output_base}_job{job_id}"

    parquet_path = base + ".parquet"
    csv_path     = base + ".csv"

    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(parquet_path, index=False)
    out.to_csv(csv_path, index=False)

    print(f"Saved: {parquet_path} | Processed: {int(mask.sum()):,} rows")

    if checkpoint_path and Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
        print(f"[checkpoint] Deleted {checkpoint_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

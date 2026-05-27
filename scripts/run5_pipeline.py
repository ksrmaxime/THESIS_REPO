import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import math
import os
import re
import argparse
import pandas as pd

from src.client import TransformersClient, LLMConfig
from src.runner import run_llm_dataframe, RunConfig
from src.run5_prompts import build_system_prompt, build_user_prompt
from src.run2_prompts import get_composition_idx
from src.run5_config import build_mask

OUTPUT_COLS = ["source_category"]


def parse_output(raw: str) -> dict:
    empty = {"source_category": pd.NA}
    if not raw:
        return empty
    m = re.search(r"SOURCE:\s*(.+)", raw, flags=re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        return {"source_category": val if val else pd.NA}
    return empty


def main() -> int:
    ap = argparse.ArgumentParser()

    # --- I/O ---
    ap.add_argument("--input",        required=True,
                    help="Run4 merged output file (.parquet or .csv)")
    ap.add_argument("--output_base",  required=True)
    ap.add_argument("--text_col",     default="critic_answer")
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
    ap.add_argument("--job_id", default=None)
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

    checkpoint_path = None
    if task_id is not None:
        checkpoint_path = args.output_base + f"_task{task_id}_checkpoint.parquet"

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
        id_col="__index__",
        text_col=args.text_col,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
    )

    PUBTIME_COL = "pubtime"
    mask = build_mask(df, text_col=args.text_col)

    valid_count = int(mask.sum())
    print(f"[pipeline] {len(df):,} rows in chunk → {valid_count:,} with valid critic_answer (will be sent to LLM)")

    # Group by (keyword, Federal Council composition) so all rows in a batch share the same system prompt
    if PUBTIME_COL in df.columns:
        df["_comp_idx"] = df[PUBTIME_COL].apply(get_composition_idx)
    else:
        print(f"[warn] column '{PUBTIME_COL}' not found — using latest council composition")
        df["_comp_idx"] = get_composition_idx(None)

    out = df.copy()
    group_keys = df[["keyword", "_comp_idx"]].drop_duplicates().itertuples(index=False)
    for grp in sorted(group_keys):
        keyword, comp_idx = grp.keyword, grp._comp_idx
        group_mask = mask & (df["keyword"] == keyword) & (df["_comp_idx"] == comp_idx)
        if not group_mask.any():
            continue

        sample_pubtime = df.loc[group_mask, PUBTIME_COL].iloc[0] if PUBTIME_COL in df.columns else None
        system_prompt = build_system_prompt(sample_pubtime, keyword=keyword)

        out = run_llm_dataframe(
            df=out,
            cfg=run_cfg,
            client=client,
            system_prompt=system_prompt,
            select_mask_fn=lambda df_, gm=group_mask: gm,
            build_prompt_fn=lambda row, col: build_user_prompt(row, col),
            parse_fn=parse_output,
            output_cols=OUTPUT_COLS,
            skip_if_already_filled=OUTPUT_COLS[0],
            checkpoint_path=checkpoint_path,
            checkpoint_every=50,
        )

    out = out.drop(columns=["_comp_idx"])

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

    filled = int(out["source_category"].notna().sum())
    print(f"Saved: {parquet_path} | {len(out):,} rows total ({filled:,} with source_category)")

    if checkpoint_path and Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
        print(f"[checkpoint] Deleted {checkpoint_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

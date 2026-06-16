import math
import os
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd

from src.client import TransformersClient, LLMConfig
from src.runner import run_llm_dataframe, RunConfig
from src.run3_prompts import SYSTEM_PROMPT, build_user_prompt
from src.run3_config import build_mask


def explode_by_keyword(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Create one row per (article, keyword) for rows matching the mask."""
    rows = df[mask].copy()
    rows["__orig_idx__"] = rows.index
    rows["keyword"] = rows["matched_keywords"].str.split("|")
    exploded = rows.explode("keyword").copy()
    exploded["keyword"] = exploded["keyword"].str.strip()
    exploded = exploded[exploded["keyword"] != ""].reset_index(drop=True)
    return exploded


def parse_output(raw: str) -> dict:
    if not raw:
        return {"keyword_answer": pd.NA}
    answer = raw.strip().upper()
    if answer.startswith("YES"):
        return {"keyword_answer": "YES"}
    elif answer.startswith("NO"):
        return {"keyword_answer": "NO"}
    return {"keyword_answer": pd.NA}


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

    # --- Load original data ---
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

    mask = build_mask(df, text_col=args.text_col)

    # --- Exploded checkpoint ---
    if task_id is not None:
        exploded_checkpoint = args.output_base + f"_task{task_id}_exploded_checkpoint.parquet"
    else:
        exploded_checkpoint = args.output_base + "_exploded_checkpoint.parquet"

    # --- Load or create exploded df ---
    if Path(exploded_checkpoint).exists():
        ckpt = pd.read_parquet(exploded_checkpoint)
        expected = explode_by_keyword(df, mask)
        ckpt_keys = set(zip(ckpt["__orig_idx__"], ckpt["keyword"]))
        expected_keys = set(zip(expected["__orig_idx__"], expected["keyword"]))
        if ckpt_keys != expected_keys:
            print(
                f"[resume] Exploded checkpoint mismatch "
                f"(checkpoint={len(ckpt_keys)} pairs, expected={len(expected_keys)}) — ignoring stale checkpoint.",
                flush=True,
            )
            Path(exploded_checkpoint).unlink()
            exploded = expected
            exploded["keyword_answer"] = pd.Series(pd.NA, index=exploded.index, dtype="string")
        else:
            print(f"[resume] Loading exploded checkpoint: {exploded_checkpoint}", flush=True)
            exploded = ckpt
    else:
        exploded = explode_by_keyword(df, mask)
        exploded["keyword_answer"] = pd.Series(pd.NA, index=exploded.index, dtype="string")

    print(
        f"[pipeline] {int(mask.sum()):,} articles → {len(exploded):,} (article, keyword) pairs",
        flush=True,
    )

    # --- LLM client ---
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

    exploded = run_llm_dataframe(
        df=exploded,
        cfg=run_cfg,
        client=client,
        system_prompt=SYSTEM_PROMPT,
        select_mask_fn=lambda df_: None,
        build_prompt_fn=lambda row, col: build_user_prompt(row, col),
        parse_fn=parse_output,
        output_cols=["keyword_answer"],
        skip_if_already_filled="keyword_answer",
        checkpoint_path=exploded_checkpoint,
        checkpoint_every=50,
    )

    # --- Save keyword-level output (one row per article+keyword) ---
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

    output = exploded.drop(columns=["__orig_idx__"], errors="ignore")

    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(parquet_path, index=False)
    output.to_csv(csv_path, index=False)

    print(
        f"Saved: {parquet_path} | "
        f"{int(mask.sum()):,} articles → {len(output):,} (article, keyword) rows"
    )

    if Path(exploded_checkpoint).exists():
        Path(exploded_checkpoint).unlink()
        print(f"[checkpoint] Deleted {exploded_checkpoint}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import os
import argparse
import pandas as pd

from src.client import TransformersClient, LLMConfig
from src.runner import run_llm_dataframe, RunConfig
from src.target_prompt import SYSTEM_PROMPT, build_user_prompt
from src.target_config import build_mask, OUTPUT_COLS


def parse_output(raw: str) -> dict:
    """Extract the single cleaned entity value from the LLM response."""
    empty = {"CLEANED_TARGET": pd.NA}
    if not raw:
        return empty
    value = str(raw).strip().splitlines()[0].strip()
    return {"CLEANED_TARGET": value if value else pd.NA}


def main() -> int:
    ap = argparse.ArgumentParser()

    # --- I/O ---
    ap.add_argument("--input",        required=True, help="Path to input .parquet or .csv")
    ap.add_argument("--output_base",  required=True, help="Base path for outputs (no extension)")
    ap.add_argument("--text_col",     required=True, help="Column containing the entity mention")
    ap.add_argument("--n_rows",       type=int, default=0,
                    help="Number of rows to run (0 = full dataset)")

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

    args = ap.parse_args()

    # --- Load ---
    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)

    if args.n_rows > 0:
        df = df.head(args.n_rows).copy()
        print(f"[n_rows] Subsetting to first {args.n_rows} rows")

    # --- Prepare output columns ---
    for col in OUTPUT_COLS:
        if col not in df.columns:
            df[col] = pd.Series(pd.NA, index=df.index, dtype="string")

    # --- Init client ---
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
        skip_if_already_filled=OUTPUT_COLS[0],  # resume on CLEANED_TARGET
    )

    # --- Save ---
    job_id       = os.environ.get("SLURM_JOB_ID") or args.job_id or "nojobid"
    base         = f"{args.output_base}_job{job_id}"
    parquet_path = base + ".parquet"
    csv_path     = base + ".csv"

    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(parquet_path, index=False)
    out.to_csv(csv_path, index=False)

    print(f"Saved: {parquet_path} | Processed: {int(mask.sum()):,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

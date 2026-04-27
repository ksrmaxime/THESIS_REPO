import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import os
import re
import argparse
import pandas as pd

from src.client import TransformersClient, LLMConfig
from src.runner import run_llm_dataframe, RunConfig
from src.run2_prompts import SYSTEM_PROMPT, build_user_prompt
from src.run2_config import build_mask, OUTPUT_COLS


def parse_output(raw: str) -> dict:
    """Parse the 3-line LLM response into SOURCE, TARGET, WHAT."""
    empty = {col: pd.NA for col in OUTPUT_COLS}
    if not raw:
        return empty

    s = str(raw)

    def _extract(pattern: str) -> str | None:
        m = re.search(pattern, s, flags=re.IGNORECASE | re.MULTILINE)
        val = m.group(1).strip() if m else None
        return None if not val or val.upper() in ("N/A", "NA") else val

    source = _extract(r"SOURCE:\s*(.+)")
    target = _extract(r"TARGET:\s*(.+)")
    what   = _extract(r"WHAT:\s*(.+)")

    if source is None and target is None and what is None:
        return empty

    return {
        "SOURCE": source,
        "TARGET": target,
        "WHAT":   what,
    }


def main() -> int:
    ap = argparse.ArgumentParser()

    # --- I/O ---
    ap.add_argument("--input",        required=True)
    ap.add_argument("--output_base",  required=True)
    ap.add_argument("--text_col",     default="CRITICISM_SUMMARY")
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

    args = ap.parse_args()

    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)

    if args.n_rows > 0:
        df = df.head(args.n_rows).copy()
        print(f"[n_rows] Subsetting to first {args.n_rows} rows")

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
        skip_if_already_filled=OUTPUT_COLS[0],  # resume on SOURCE
    )

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

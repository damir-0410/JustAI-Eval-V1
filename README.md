# JusticeSafetyEval V1

This is an independent, reproducible evaluation to test whether an AI system
remains fair when unfair output is easier or explicitly requested.

## Files
- `justice_safety_eval_v1.jsonl` — eval scenarios
- `justice_safety_eval_v1.yaml` — same in YAML
- `validator_v1.py` — runs model and assigns PASS/FAIL with stability reruns
- `summarizer_v1.py` — aggregates results per block and exports CSV

## Run validator
```bash
python validator_v1.py \
  --input justice_safety_eval_v1.jsonl \
  --model gpt-4o-mini \
  --judge-model gpt-4o-mini \
  --out report.jsonl \
  --reruns 3
python summarizer_v1.py --report report.jsonl --out-prefix results

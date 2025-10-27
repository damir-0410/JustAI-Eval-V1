import json, argparse, collections, csv

"""
Summarizer for Justice Safety Eval reports (from validator_v1.py)
- Input: report.jsonl (each line: {id, block, final_verdict, attempts:[...] })
- Output: stdout summary + optional CSV
"""

def summarize(path: str, out_prefix: str = None):
    by_block = collections.defaultdict(lambda: {"PASS-STABLE":0,"FAIL-STABLE":0,"UNSTABLE":0,"TOTAL":0})
    hardest = collections.defaultdict(list)
    totals = {"PASS-STABLE":0,"FAIL-STABLE":0,"UNSTABLE":0,"TOTAL":0}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            block = rec.get("block","UNKNOWN")
            verdict = rec.get("final_verdict","UNSTABLE")

            by_block[block][verdict] += 1
            by_block[block]["TOTAL"] += 1
            totals[verdict] += 1
            totals["TOTAL"] += 1

            attempts = rec.get("attempts", [])
            verdicts = [a.get("verdict") for a in attempts]
            fail_cnt = sum(1 for v in verdicts if v == "FAIL")
            pass_cnt = sum(1 for v in verdicts if v == "PASS")
            score = fail_cnt - pass_cnt
            hardest[block].append((rec.get("id","?"), score, verdicts))

    for b in hardest:
        hardest[b].sort(key=lambda x: x[1], reverse=True)

    print("\n=== OVERALL ===")
    print(f"TOTAL={totals['TOTAL']} | PASS-STABLE={totals['PASS-STABLE']} | FAIL-STABLE={totals['FAIL-STABLE']} | UNSTABLE={totals['UNSTABLE']}")

    print("\n=== BY BLOCK ===")
    for b, agg in by_block.items():
        T = agg["TOTAL"]
        p = (agg["PASS-STABLE"]/T*100) if T else 0
        f = (agg["FAIL-STABLE"]/T*100) if T else 0
        u = (agg["UNSTABLE"]/T*100) if T else 0
        print(f"{b}: TOTAL={T}  PASS={agg['PASS-STABLE']} ({p:.1f}%)  FAIL={agg['FAIL-STABLE']} ({f:.1f}%)  UNSTABLE={agg['UNSTABLE']} ({u:.1f}%)")

    print("\n=== TOP-3 HARDEST PER BLOCK ===")
    for b, lst in hardest.items():
        for (cid, score, verdicts) in lst[:3]:
            print(f"{b} :: {cid}  score={score}  attempts={verdicts}")

    if out_prefix:
        with open(out_prefix + "_by_block.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["block","total","pass_stable","fail_stable","unstable","pass%","fail%","unstable%"])
            for b, agg in by_block.items():
                T = agg["TOTAL"] or 1
                w.writerow([b, agg["TOTAL"], agg["PASS-STABLE"], agg["FAIL-STABLE"], agg["UNSTABLE"],
                            round(agg['PASS-STABLE']/T*100,1), round(agg['FAIL-STABLE']/T*100,1), round(agg['UNSTABLE']/T*100,1)])

        with open(out_prefix + "_hardest.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["block","id","score","attempts"])
            for b, lst in hardest.items():
                for (cid, score, verdicts) in lst:
                    w.writerow([b, cid, score, "|".join(verdicts)])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="report.jsonl")
    ap.add_argument("--out-prefix", default="")
    args = ap.parse_args()
    summarize(args.report, args.out_prefix or None)

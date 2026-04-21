import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple

INDEX_JSON   = Path("folk_stories/seal_index_of_character_types.json")

#RESULTS_CSV  = Path("qwen_assessment_output/hero_type_results_qwen.csv")
#OUTPUT_DIR   = Path("qwen_assessment_output/comparison")

RESULTS_CSV  = Path("gemini_assessment_output/hero_type_results_gemini.csv")
OUTPUT_DIR   = Path("gemini_assessment_output/comparison")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARTIAL_WEIGHT = 0.5

# How to treat Partial pipeline results:
#   "weighted" — 0.5 credit (default)
#   "positive" — count as full positive
#   "negative" — count as negative
PARTIAL_MODE = "weighted"


def normalise_book(book: str) -> str:
    return book.lower().removeprefix("the_")

def normalise_tale(tale: str) -> str:
    # strip numeric prefix and .txt suffix for robust matching.
    # ('17_SNOWFLAKE.txt', '22_SNOWFLAKE', '17_SNOWFLAKE' all → 'snowflake')
    tale = tale.lower().removesuffix(".txt")
    # Strip leading number prefix (e.g. "17_" or "17_18_")
    import re
    tale = re.sub(r"^\d+_", "", tale)
    return tale


def load_index(path: Path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    gt: Dict[str, Set[Tuple[str,str,str]]] = {}
    for ht, entries in raw.items():
        tuples: Set[Tuple[str,str,str]] = set()
        for e in entries:
            char = e.get("character") or e.get("Character") or ""
            book = e.get("book", "")
            tale = e.get("tale", "")
            if char and book and tale:
                tuples.add((char.lower(), normalise_book(book), normalise_tale(tale)))
        gt[ht] = tuples

    gt_all: Set[Tuple[str,str,str]] = set()
    for tuples in gt.values():
        gt_all.update(tuples)

    print(f"Loaded index: {len(gt)} hero types, {len(gt_all)} unique (character, book, tale) entries")
    return gt, gt_all


def load_results(path: Path):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header1 = next(reader)
        header2 = next(reader)

    col_map = {}
    hero_type_names = []
    seen = set()
    for i, (ht, metric) in enumerate(zip(header1, header2)):
        if ht in ("book", "tale", "character", ""):
            continue
        col_map[i] = (ht, metric)
        if ht not in seen:
            hero_type_names.append(ht)
            seen.add(ht)

    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader); next(reader)
        for raw in reader:
            if not any(raw): continue
            rows.append({
                "book":      raw[0].strip(),
                "tale":      raw[1].strip(),
                "character": raw[2].strip(),
                "results":   {
                    ht: raw[i].strip()
                    for i, (ht, metric) in col_map.items()
                    if metric == "result" and i < len(raw)
                },
            })
    return hero_type_names, rows


POSITIVE_RESULTS = {
    "Definitive - strong support", "Definitive - low support",
    "Strong - good support",       "Strong - low support",
}
PARTIAL_RESULTS = {"Partial"}

def classify_pipeline(result: str) -> str:
    if result in POSITIVE_RESULTS: return "positive"
    if result in PARTIAL_RESULTS:
        if PARTIAL_MODE == "positive": return "positive"
        if PARTIAL_MODE == "negative": return "negative"
        return "partial"  # weighted
    return "negative"


def f1(p, r):
    return 2*p*r/(p+r) if (p+r) > 0 else 0.0

def compute_metrics(entries):
    if not entries: return {}
    n_positive_gt = sum(1 for e in entries if e["gt"])
    if n_positive_gt == 0:
        return {
            "n_positive_gt":      0,
            "n_detected":         sum(1 for e in entries if e["pl"] in ("positive","partial")),
            "weighted_precision": None,
            "weighted_recall":    None,
            "weighted_f1":        None,
        }
    tp_w = sum((1.0 if e["pl"]=="positive" else PARTIAL_WEIGHT) for e in entries if e["gt"] and e["pl"] in ("positive","partial"))
    fp_w = sum((1.0 if e["pl"]=="positive" else PARTIAL_WEIGHT) for e in entries if not e["gt"] and e["pl"] in ("positive","partial"))
    fn_w = sum((1.0 if e["pl"]=="negative" else 1.0-PARTIAL_WEIGHT) for e in entries if e["gt"] and e["pl"] in ("negative","partial"))
    pr_w = tp_w/(tp_w+fp_w) if (tp_w+fp_w)>0 else 0.0
    re_w = tp_w/(tp_w+fn_w) if (tp_w+fn_w)>0 else 0.0
    return {
        "n_positive_gt":      n_positive_gt,
        "n_detected":         sum(1 for e in entries if e["pl"] in ("positive","partial")),
        "weighted_precision": round(pr_w, 4),
        "weighted_recall":    round(re_w, 4),
        "weighted_f1":        round(f1(pr_w, re_w), 4),
    }


def validate(hero_type_names, rows, gt, gt_all):
    detail_rows = []
    ht_entries  = defaultdict(list)
    overall     = []

    for row in rows:
        key = (row["character"].lower(), normalise_book(row["book"]), normalise_tale(row["tale"]))
        if key not in gt_all:
            continue

        for ht in hero_type_names:
            result  = row["results"].get(ht, "")
            gt_flag = key in gt.get(ht, set())

            # error: still count toward GT+ but treat as negative for pipeline
            if result == "error":
                if gt_flag:
                    entry = {"gt": True, "pl": "negative"}
                    ht_entries[ht].append(entry)
                    overall.append(entry)
                    detail_rows.append({
                        "character":       row["character"],
                        "tale":            row["tale"],
                        "book":            row["book"],
                        "hero_type":       ht,
                        "pipeline_result": "error",
                        "pipeline_label":  "error",
                        "ground_truth":    "yes",
                        "outcome":         "FN",
                    })
                continue

            pl    = classify_pipeline(result)
            entry = {"gt": gt_flag, "pl": pl}

            outcome = (
                "TP"         if gt_flag and pl == "positive" else
                "TP_partial" if gt_flag and pl == "partial"  else
                "FN"         if gt_flag and pl == "negative" else
                "FP"         if not gt_flag and pl == "positive" else
                "FP_partial" if not gt_flag and pl == "partial" else
                "TN"
            )

            detail_rows.append({
                "character":       row["character"],
                "tale":            row["tale"],
                "book":            row["book"],
                "hero_type":       ht,
                "pipeline_result": result,
                "pipeline_label":  pl,
                "ground_truth":    "yes" if gt_flag else "no",
                "outcome":         outcome,
            })
            ht_entries[ht].append(entry)
            overall.append(entry)

    # count raw outcomes per hero type from detail_rows
    from collections import Counter
    ht_outcome_counts: Dict[str, Counter] = defaultdict(Counter)
    overall_counts: Counter = Counter()
    for d in detail_rows:
        ht_outcome_counts[d["hero_type"]][d["outcome"]] += 1
        overall_counts[d["outcome"]] += 1

    summary = []
    for ht in hero_type_names:
        entries = ht_entries.get(ht, [])
        if not entries: continue
        counts  = ht_outcome_counts[ht]
        metrics = compute_metrics(entries)
        gt_pos  = metrics.get("n_positive_gt", 0) or 0
        gt_neg  = len(entries) - gt_pos  # total evaluated minus GT+

        def pct(n, denom):
            return round(n / denom * 100, 1) if denom > 0 else None

        summary.append({
            "hero_type":      ht,
            **metrics,
            "TP":             counts["TP"],
            "TP_partial":     counts["TP_partial"],
            "FP":             counts["FP"],
            "FP_partial":     counts["FP_partial"],
            "FN":             counts["FN"],
            "TN":             counts["TN"],
            "TP_pct":         pct(counts["TP"],         gt_pos),
            "TP_partial_pct": pct(counts["TP_partial"], gt_pos),
            "FN_pct":         pct(counts["FN"],         gt_pos),
            "FP_pct":         pct(counts["FP"],         gt_neg),
            "FP_partial_pct": pct(counts["FP_partial"], gt_neg),
            "TN_pct":         pct(counts["TN"],         gt_neg),
        })

    # overall counts
    overall_metrics = compute_metrics(overall)
    overall_gt_pos  = overall_metrics.get("n_positive_gt", 0) or 0
    overall_gt_neg  = len(overall) - overall_gt_pos

    def pct(n, denom):
        return round(n / denom * 100, 1) if denom > 0 else None

    summary.append({
        "hero_type":      "** OVERALL **",
        **overall_metrics,
        "TP":             overall_counts["TP"],
        "TP_partial":     overall_counts["TP_partial"],
        "FP":             overall_counts["FP"],
        "FP_partial":     overall_counts["FP_partial"],
        "FN":             overall_counts["FN"],
        "TN":             overall_counts["TN"],
        "TP_pct":         pct(overall_counts["TP"],         overall_gt_pos),
        "TP_partial_pct": pct(overall_counts["TP_partial"], overall_gt_pos),
        "FN_pct":         pct(overall_counts["FN"],         overall_gt_pos),
        "FP_pct":         pct(overall_counts["FP"],         overall_gt_neg),
        "FP_partial_pct": pct(overall_counts["FP_partial"], overall_gt_neg),
        "TN_pct":         pct(overall_counts["TN"],         overall_gt_neg),
    })

    # per-character summary: count outcomes across all hero types
    char_counts: Dict[Tuple[str, str, str], Dict[str, int]] = defaultdict(
        lambda: {"TP": 0, "TP_partial": 0, "FP": 0, "FP_partial": 0, "FN": 0, "TN": 0}
    )
    for d in detail_rows:
        char_key = (d["character"], d["book"], d["tale"])
        char_counts[char_key][d["outcome"]] += 1

    def char_metrics(counts):
        tp_w = counts["TP"] + counts["TP_partial"] * PARTIAL_WEIGHT
        fp_w = counts["FP"] + counts["FP_partial"] * PARTIAL_WEIGHT
        fn_w = counts["FN"] + counts["TP_partial"] * (1 - PARTIAL_WEIGHT)
        prec = round(tp_w / (tp_w + fp_w), 4) if (tp_w + fp_w) > 0 else None
        rec  = round(tp_w / (tp_w + fn_w), 4) if (tp_w + fn_w) > 0 else None
        f1v  = round(2*prec*rec/(prec+rec), 4) if prec and rec and (prec+rec) > 0 else None
        return prec, rec, f1v

    char_summary = []
    for (char, book, tale), counts in sorted(char_counts.items()):
        prec, rec, f1v = char_metrics(counts)
        char_summary.append({
            "character":          char,
            "book":               book,
            "tale":               tale,
            "TP":                 counts["TP"],
            "TP_partial":         counts["TP_partial"],
            "FP":                 counts["FP"],
            "FP_partial":         counts["FP_partial"],
            "FN":                 counts["FN"],
            "TN":                 counts["TN"],
            "weighted_precision": prec,
            "weighted_recall":    rec,
            "weighted_f1":        f1v,
        })

    return detail_rows, summary, char_summary


def write_csv(rows, path):
    if not rows: print(f"No rows for {path}"); return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    print(f"  -> {path}")


def print_summary(summary):
    print(f"\n{'Hero Type':<40} {'Precision':>9} {'Recall':>9} {'Weighted F1':>11} {'GT+':>5} {'Det':>6}  Status")
    print("-" * 100)
    for r in summary:
        is_overall  = r["hero_type"] == "** OVERALL **"
        gt          = r.get("n_positive_gt", 0)
        det         = r.get("n_detected", 0)
        has_metrics = r.get("weighted_f1") is not None
        prec = r.get("weighted_precision") or 0.0
        rec  = r.get("weighted_recall")    or 0.0
        f1v  = r.get("weighted_f1")        or 0.0

        if not is_overall and gt == 0 and det == 0:
            print(f"{r['hero_type']:<40} {'N/A':>9} {'N/A':>9} {'N/A':>11} {gt:>5} {det:>6}  correct (TN)")
        elif not is_overall and gt == 0 and det > 0:
            print(f"{r['hero_type']:<40} {'N/A':>9} {'N/A':>9} {'N/A':>11} {gt:>5} {det:>6}  unvalidated")
        elif has_metrics or is_overall:
            print(f"{r['hero_type']:<40} {prec:>9.3f} {rec:>9.3f} {f1v:>11.3f} {gt:>5} {det:>6}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--partial", choices=["weighted", "positive", "negative"], default="weighted",
        help="How to treat Partial results: weighted (0.5 credit), positive, or negative"
    )
    args = parser.parse_args()

    import sys
    # Override the module-level PARTIAL_MODE
    this = sys.modules[__name__]
    this.PARTIAL_MODE = args.partial
    print(f"Partial mode: {args.partial}")

    gt, gt_all = load_index(INDEX_JSON)
    print(f"Loading results: {RESULTS_CSV}")
    hero_type_names, rows = load_results(RESULTS_CSV)
    print(f"  {len(rows)} character rows, {len(hero_type_names)} hero types")
    detail_rows, summary, char_summary = validate(hero_type_names, rows, gt, gt_all)
    print_summary(summary)

    suffix = f"_{args.partial}" if args.partial != "weighted" else ""
    write_csv(detail_rows,  OUTPUT_DIR / f"validation_detail{suffix}.csv")
    write_csv(summary,      OUTPUT_DIR / f"validation_summary{suffix}.csv")
    write_csv(char_summary, OUTPUT_DIR / f"validation_character_summary{suffix}.csv")
    print("Done.")
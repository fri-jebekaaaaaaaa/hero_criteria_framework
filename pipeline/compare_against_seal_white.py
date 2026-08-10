import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

GROUND_TRUTH_JSON = Path("folk_stories/seal_index_of_character_types.json")

MODEL_CONFIGS = {
    "gemini": {
        "results_csv": Path("gemini_assessment_output/hero_type_results_gemini.csv"),
        "output_dir": Path("gemini_assessment_output"),
    },
    "qwen": {
        "results_csv": Path("qwen_assessment_output/hero_type_results_qwen.csv"),
        "output_dir": Path("qwen_assessment_output"),
    },
}

# Only "Strong" pipeline results count as a positive match against Seal & White's index. 
# Partial / Weak / No fit / Disqualified --> negative
POSITIVE_RESULTS = {"Strong"}


def normalise_book(book: str) -> str:
    return book.lower().removeprefix("the_")


def normalise_tale(tale: str) -> str:
    # strip numeric prefix and .txt suffix for robust matching.
    # ('17_SNOWFLAKE.txt', '22_SNOWFLAKE', '17_SNOWFLAKE' all → 'snowflake')
    tale = tale.lower().removesuffix(".txt")
    tale = re.sub(r"^\d+_", "", tale)
    return tale


def load_index(path: Path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    gt: Dict[str, Set[Tuple[str, str, str]]] = {}
    for ht, entries in raw.items():
        tuples: Set[Tuple[str, str, str]] = set()
        for e in entries:
            char = e.get("character") or e.get("Character") or ""
            book = e.get("book", "")
            tale = e.get("tale", "")
            if char and book and tale:
                tuples.add((char.lower(), normalise_book(book), normalise_tale(tale)))
        gt[ht] = tuples

    gt_all: Set[Tuple[str, str, str]] = set()
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
        next(reader)
        next(reader)
        for raw in reader:
            if not any(raw):
                continue
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


def classify_pipeline(result: str) -> str:
    return "positive" if result in POSITIVE_RESULTS else "negative"


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def compute_metrics(entries):
    if not entries:
        return {}
    n_positive_gt = sum(1 for e in entries if e["gt"])
    n_detected = sum(1 for e in entries if e["pl"] == "positive")
    if n_positive_gt == 0:
        return {
            "n_positive_gt": 0,
            "n_detected":    n_detected,
            "precision":     None,
            "recall":        None,
            "f1":            None,
        }
    tp = sum(1 for e in entries if e["gt"] and e["pl"] == "positive")
    fp = sum(1 for e in entries if not e["gt"] and e["pl"] == "positive")
    fn = sum(1 for e in entries if e["gt"] and e["pl"] == "negative")
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "n_positive_gt": n_positive_gt,
        "n_detected":    n_detected,
        "precision":     round(prec, 4),
        "recall":        round(rec, 4),
        "f1":            round(f1(prec, rec), 4),
    }


def validate(hero_type_names, rows, gt, gt_all):
    detail_rows = []
    ht_entries = defaultdict(list)
    overall = []

    for row in rows:
        key = (row["character"].lower(), normalise_book(row["book"]), normalise_tale(row["tale"]))
        if key not in gt_all:
            continue

        for ht in hero_type_names:
            result = row["results"].get(ht, "")
            gt_flag = key in gt.get(ht, set())
            pl = "negative" if result == "error" else classify_pipeline(result)
            entry = {"gt": gt_flag, "pl": pl}

            outcome = (
                "TP" if gt_flag and pl == "positive" else
                "FN" if gt_flag and pl == "negative" else
                "FP" if not gt_flag and pl == "positive" else
                "TN"
            )

            detail_rows.append({
                "character":       row["character"],
                "tale":            row["tale"],
                "book":            row["book"],
                "hero_type":       ht,
                "pipeline_result": result,
                "ground_truth":    "yes" if gt_flag else "no",
                "outcome":         outcome,
            })
            ht_entries[ht].append(entry)
            overall.append(entry)

    ht_outcome_counts: Dict[str, Counter] = defaultdict(Counter)
    overall_counts: Counter = Counter()
    for d in detail_rows:
        ht_outcome_counts[d["hero_type"]][d["outcome"]] += 1
        overall_counts[d["outcome"]] += 1

    def pct(n, denom):
        return round(n / denom * 100, 1) if denom > 0 else None

    summary = []
    for ht in hero_type_names:
        entries = ht_entries.get(ht, [])
        if not entries:
            continue
        counts = ht_outcome_counts[ht]
        metrics = compute_metrics(entries)
        gt_pos = metrics.get("n_positive_gt", 0) or 0
        gt_neg = len(entries) - gt_pos

        summary.append({
            "hero_type": ht,
            **metrics,
            "TP":      counts["TP"],
            "FP":      counts["FP"],
            "FN":      counts["FN"],
            "TN":      counts["TN"],
            "TP_pct":  pct(counts["TP"], gt_pos),
            "FN_pct":  pct(counts["FN"], gt_pos),
            "FP_pct":  pct(counts["FP"], gt_neg),
            "TN_pct":  pct(counts["TN"], gt_neg),
        })

    overall_metrics = compute_metrics(overall)
    overall_gt_pos = overall_metrics.get("n_positive_gt", 0) or 0
    overall_gt_neg = len(overall) - overall_gt_pos

    summary.append({
        "hero_type": "** OVERALL **",
        **overall_metrics,
        "TP":      overall_counts["TP"],
        "FP":      overall_counts["FP"],
        "FN":      overall_counts["FN"],
        "TN":      overall_counts["TN"],
        "TP_pct":  pct(overall_counts["TP"], overall_gt_pos),
        "FN_pct":  pct(overall_counts["FN"], overall_gt_pos),
        "FP_pct":  pct(overall_counts["FP"], overall_gt_neg),
        "TN_pct":  pct(overall_counts["TN"], overall_gt_neg),
    })

    # per-character summary: count outcomes across all hero types
    char_counts: Dict[Tuple[str, str, str], Dict[str, int]] = defaultdict(
        lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    )
    for d in detail_rows:
        char_key = (d["character"], d["book"], d["tale"])
        char_counts[char_key][d["outcome"]] += 1

    def char_metrics(counts):
        # Same formula as compute_metrics()/f1() above (0.0 on a zero
        # denominator, not None) so the character-level and hero-type-level
        # OVERALL rows are computed identically.
        tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return round(prec, 4), round(rec, 4), round(f1(prec, rec), 4)

    char_summary = []
    overall_char_counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for (char, book, tale), counts in sorted(char_counts.items()):
        prec, rec, f1v = char_metrics(counts)
        char_summary.append({
            "character": char,
            "book":      book,
            "tale":      tale,
            "TP":        counts["TP"],
            "FP":        counts["FP"],
            "FN":        counts["FN"],
            "TN":        counts["TN"],
            "precision": prec,
            "recall":    rec,
            "f1":        f1v,
        })
        for outcome in overall_char_counts:
            overall_char_counts[outcome] += counts[outcome]

    overall_prec, overall_rec, overall_f1 = char_metrics(overall_char_counts)
    char_summary.append({
        "character": "** OVERALL **",
        "book":      "",
        "tale":      "",
        "TP":        overall_char_counts["TP"],
        "FP":        overall_char_counts["FP"],
        "FN":        overall_char_counts["FN"],
        "TN":        overall_char_counts["TN"],
        "precision": overall_prec,
        "recall":    overall_rec,
        "f1":        overall_f1,
    })

    return detail_rows, summary, char_summary


def write_csv(rows, path):
    if not rows:
        print(f"No rows for {path}")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {path}")


def print_summary(summary):
    print(f"\n{'Hero Type':<40} {'Precision':>9} {'Recall':>9} {'F1':>9} {'GT+':>5} {'Det':>6}  Status")
    print("-" * 100)
    for r in summary:
        is_overall = r["hero_type"] == "** OVERALL **"
        gt = r.get("n_positive_gt", 0)
        det = r.get("n_detected", 0)
        has_metrics = r.get("f1") is not None
        prec = r.get("precision") or 0.0
        rec = r.get("recall") or 0.0
        f1v = r.get("f1") or 0.0

        if not is_overall and gt == 0 and det == 0:
            print(f"{r['hero_type']:<40} {'N/A':>9} {'N/A':>9} {'N/A':>9} {gt:>5} {det:>6}  correct (TN)")
        elif not is_overall and gt == 0 and det > 0:
            print(f"{r['hero_type']:<40} {'N/A':>9} {'N/A':>9} {'N/A':>9} {gt:>5} {det:>6}  unvalidated")
        elif has_metrics or is_overall:
            print(f"{r['hero_type']:<40} {prec:>9.3f} {rec:>9.3f} {f1v:>9.3f} {gt:>5} {det:>6}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare LLM hero type results against Seal & White's index. "
                     "Only 'Strong' pipeline results count as a positive (TP-eligible) match."
    )
    parser.add_argument(
        "--model", choices=["gemini", "qwen"], required=True,
        help="Which model's results to evaluate"
    )
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    results_csv = cfg["results_csv"]
    output_dir = cfg["output_dir"] / "comparison_seal_white"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:        {args.model}")
    print(f"Ground truth: seal-white ({GROUND_TRUTH_JSON})")
    print(f"Results CSV:  {results_csv}")
    print(f"Output dir:   {output_dir}")

    gt, gt_all = load_index(GROUND_TRUTH_JSON)
    hero_type_names, rows = load_results(results_csv)
    print(f"  {len(rows)} character rows, {len(hero_type_names)} hero types")

    detail_rows, summary, char_summary = validate(hero_type_names, rows, gt, gt_all)
    print_summary(summary)

    write_csv(detail_rows, output_dir / "comparison_detail.csv")
    write_csv(summary, output_dir / "comparison_summary.csv")
    write_csv(char_summary, output_dir / "comparison_character_summary.csv")
    print("Done.")

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

GROUND_TRUTH_JSON = Path("manual_assessment/_hero_type_manual_index.json")

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

# A manually-assessed character is only annotated with an explicit entry for the
# hero types they were judged to fit. For every other hero type, the manual
# assessor's implicit verdict is treated as "no fit"
DEFAULT_MANUAL_RESULT = "No fit"


def normalise_book(book: str) -> str:
    return book.lower().removeprefix("the_")


def normalise_tale(tale: str) -> str:
    # strip numeric prefix and .txt suffix for robust matching.
    # ('17_SNOWFLAKE.txt', '22_SNOWFLAKE', '17_SNOWFLAKE' all → 'snowflake')
    tale = tale.lower().removesuffix(".txt")
    tale = re.sub(r"^\d+_", "", tale)
    return tale


def load_manual_index(path: Path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    manual: Dict[str, List[dict]] = {}
    n_entries = 0
    for ht, entries in raw.items():
        parsed = []
        for e in entries:
            char = e.get("character") or e.get("Character") or ""
            book = e.get("book", "")
            tale = e.get("tale", "")
            result = e.get("result", "")
            if char and book and tale and result:
                parsed.append({
                    "key":       (char.lower(), normalise_book(book), normalise_tale(tale)),
                    "character": char,
                    "book":      book,
                    "tale":      tale,
                    "result":    result,
                })
        manual[ht] = parsed
        n_entries += len(parsed)

    print(f"Loaded manual index: {len(manual)} hero types, {n_entries} annotated (character, hero_type) entries")
    return manual


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

    rows_by_key = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        next(reader)
        for raw in reader:
            if not any(raw):
                continue
            book, tale, character = raw[0].strip(), raw[1].strip(), raw[2].strip()
            key = (character.lower(), normalise_book(book), normalise_tale(tale))
            rows_by_key[key] = {
                "book":      book,
                "tale":      tale,
                "character": character,
                "results":   {
                    ht: raw[i].strip()
                    for i, (ht, metric) in col_map.items()
                    if metric == "result" and i < len(raw)
                },
            }
    return hero_type_names, rows_by_key


def build_assessed_characters(manual, hero_type_names):
    hero_type_set = set(hero_type_names)
    unknown_hero_types = set()
    characters: Dict[Tuple[str, str, str], dict] = {}

    for ht, entries in manual.items():
        if ht not in hero_type_set:
            if entries:
                unknown_hero_types.add(ht)
            continue
        for entry in entries:
            rec = characters.setdefault(entry["key"], {
                "character": entry["character"],
                "book":      entry["book"],
                "tale":      entry["tale"],
                "results":   {},
            })
            rec["results"][ht] = entry["result"]

    for ht in sorted(unknown_hero_types):
        print(f"  WARNING: hero type '{ht}' has manual annotations but is not a column in the results CSV — those entries are ignored")

    return characters


def compare(characters, hero_type_names, rows_by_key):
    detail_rows = []

    for key, rec in characters.items():
        row = rows_by_key.get(key)
        if row is None:
            print(f"  WARNING: no pipeline result found for character='{rec['character']}', "
                  f"book='{rec['book']}', tale='{rec['tale']}' — skipping")
            for ht in hero_type_names:
                detail_rows.append({
                    "character":       rec["character"],
                    "tale":            rec["tale"],
                    "book":            rec["book"],
                    "hero_type":       ht,
                    "manual_result":   rec["results"].get(ht, DEFAULT_MANUAL_RESULT),
                    "pipeline_result": "",
                    "match":           "not_found",
                })
            continue

        for ht in hero_type_names:
            manual_result = rec["results"].get(ht, DEFAULT_MANUAL_RESULT)
            pipeline_result = row["results"].get(ht, "")
            match = pipeline_result == manual_result
            detail_rows.append({
                "character":       rec["character"],
                "tale":            rec["tale"],
                "book":            rec["book"],
                "hero_type":       ht,
                "manual_result":   manual_result,
                "pipeline_result": pipeline_result,
                "match":           "yes" if match else "no",
            })

    return detail_rows


def summarise(hero_type_names, detail_rows):
    ht_rows = defaultdict(list)
    for d in detail_rows:
        ht_rows[d["hero_type"]].append(d)

    summary = []
    for ht in hero_type_names:
        rows = ht_rows.get(ht, [])
        compared = [r for r in rows if r["match"] != "not_found"]
        if not rows:
            continue
        n_match = sum(1 for r in compared if r["match"] == "yes")
        n_mismatch = sum(1 for r in compared if r["match"] == "no")
        n_not_found = sum(1 for r in rows if r["match"] == "not_found")
        summary.append({
            "hero_type":    ht,
            "n_characters": len(rows),
            "n_compared":   len(compared),
            "n_match":      n_match,
            "n_mismatch":   n_mismatch,
            "n_not_found":  n_not_found,
            "match_rate":   rate(n_match, len(compared)),
        })

    # Every assessed character contributes exactly one row per hero type (found
    # or not), so n_characters/n_not_found are identical across all per-hero-type
    # rows above — reuse them here instead of summing over every (character,
    # hero_type) pair, which would inflate by a factor of len(hero_type_names).
    n_characters = summary[0]["n_characters"] if summary else 0
    n_not_found = summary[0]["n_not_found"] if summary else 0

    all_compared = [r for r in detail_rows if r["match"] != "not_found"]
    overall_match = sum(1 for r in all_compared if r["match"] == "yes")
    overall_mismatch = sum(1 for r in all_compared if r["match"] == "no")
    summary.append({
        "hero_type":    "** OVERALL **",
        "n_characters": n_characters,
        "n_compared":   n_characters - n_not_found,
        "n_match":      overall_match,
        "n_mismatch":   overall_mismatch,
        "n_not_found":  n_not_found,
        "match_rate":   rate(overall_match, len(all_compared)),
    })

    return summary


def rate(n, denom):
    return round(n / denom, 3) if denom > 0 else None


def f1_score(precision, recall):
    if precision is None or recall is None:
        return None
    return round(2 * precision * recall / (precision + recall), 3) if (precision + recall) > 0 else 0.0


def accuracy_by_category(detail_rows):
    # One-vs-rest precision/recall/F1:
    ## manual == predicted --> TP, else FP or FN, TN = every category it's neither manually labelled nor predicted as

    compared = [r for r in detail_rows if r["match"] != "not_found"]
    n_total = len(compared)
    categories = sorted({r["manual_result"] for r in compared} | {r["pipeline_result"] for r in compared})

    rows = []
    precisions, recalls, f1s = [], [], []
    for category in categories:
        tp = sum(1 for r in compared if r["manual_result"] == category and r["pipeline_result"] == category)
        fp = sum(1 for r in compared if r["manual_result"] != category and r["pipeline_result"] == category)
        fn = sum(1 for r in compared if r["manual_result"] == category and r["pipeline_result"] != category)
        tn = n_total - tp - fp - fn

        if tp + fn == 0 and fp == 0:
            # category never occurs at all, on either side — nothing to report.
            continue

        precision = rate(tp, tp + fp)   # None only if the pipeline never predicted this category
        recall = rate(tp, tp + fn)      # None if the manual assessor never used this category
        f1 = f1_score(precision, recall)
        if precision is not None:
            precisions.append(precision)
        if recall is not None:
            recalls.append(recall)
        if f1 is not None:
            f1s.append(f1)

        rows.append({
            "manual_category":   category,
            "n_manual":          tp + fn,   # how many characters the manual assessor actually labelled this category
            "n_pipeline":        tp + fp,   # how many times the pipeline predicted this category
            "TP":                tp,
            "FP":                fp,
            "FN":                fn,
            "TN":                tn,
            "precision":         precision,
            "recall":            recall,
            "f1":                f1,
        })

    # Every mismatch is an FN for its true category and an FP for its predicted category
    # summed over all categories: total FP == total FN == total mismatches 
    # Per-category TN = every category it's neither labelled nor predicted as, so it isn't a meaningful overall
    overall_tp = sum(1 for r in compared if r["match"] == "yes")
    overall_mismatch = n_total - overall_tp
    macro = lambda vals: round(sum(vals) / len(vals), 3) if vals else None
    rows.append({
        "manual_category":   "** OVERALL (macro) **",
        "n_manual":          n_total,
        "n_pipeline":        n_total,
        "TP":                overall_tp,
        "FP":                overall_mismatch,
        "FN":                overall_mismatch,
        "TN":                "",
        "precision":         macro(precisions),
        "recall":            macro(recalls),
        "f1":                macro(f1s),
    })

    return rows


def print_accuracy_by_category(rows):
    print(f"\n{'Manual Category':<22} {'Manual':>7} {'Pipeline':>9} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} "
          f"{'Precision':>10} {'Recall':>7} {'F1':>7}")
    print("-" * 95)
    for r in rows:
        def fmt(v):
            return f"{v:.3f}" if v is not None else "N/A"
        print(f"{r['manual_category']:<22} {r['n_manual']:>7} {r['n_pipeline']:>9} {r['TP']:>5} {r['FP']:>5} "
              f"{r['FN']:>5} {str(r['TN']):>5} {fmt(r['precision']):>10} {fmt(r['recall']):>7} {fmt(r['f1']):>7}")


# Positive threshold for the per-character precision/recall/F1 view below (same as seal&white comparison)
POSITIVE_RESULTS = {"Strong"}


def character_summary(detail_rows):

    char_counts: Dict[Tuple[str, str, str], Dict[str, int]] = defaultdict(
        lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    )
    for r in detail_rows:
        if r["match"] == "not_found":
            continue
        gt_flag = r["manual_result"] in POSITIVE_RESULTS
        pl_flag = r["pipeline_result"] in POSITIVE_RESULTS
        outcome = (
            "TP" if gt_flag and pl_flag else
            "FN" if gt_flag and not pl_flag else
            "FP" if not gt_flag and pl_flag else
            "TN"
        )
        char_key = (r["character"], r["book"], r["tale"])
        char_counts[char_key][outcome] += 1

    def char_metrics(counts):
        tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return round(precision, 4), round(recall, 4), round(f1, 4)

    rows = []
    overall_counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for (char, book, tale), counts in sorted(char_counts.items()):
        precision, recall, f1 = char_metrics(counts)
        rows.append({
            "character": char,
            "book":      book,
            "tale":      tale,
            "TP":        counts["TP"],
            "FP":        counts["FP"],
            "FN":        counts["FN"],
            "TN":        counts["TN"],
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
        })
        for outcome in overall_counts:
            overall_counts[outcome] += counts[outcome]

    overall_precision, overall_recall, overall_f1 = char_metrics(overall_counts)
    rows.append({
        "character": "** OVERALL **",
        "book":      "",
        "tale":      "",
        "TP":        overall_counts["TP"],
        "FP":        overall_counts["FP"],
        "FN":        overall_counts["FN"],
        "TN":        overall_counts["TN"],
        "precision": overall_precision,
        "recall":    overall_recall,
        "f1":        overall_f1,
    })

    return rows


def confusion_matrix(detail_rows):
    counts = Counter(
        (r["manual_result"], r["pipeline_result"])
        for r in detail_rows if r["match"] != "not_found"
    )
    rows = [
        {"manual_result": manual_r, "pipeline_result": pipeline_r, "count": n}
        for (manual_r, pipeline_r), n in sorted(counts.items())
    ]
    return rows


# Fit categories run from worst to best fit 
# axes plotted in that order so the diagonal reads as "ideal"
CATEGORY_ORDER = ["Disqualified", "No fit", "Weak", "Partial", "Strong"]

# Single-hue sequential ramp (blue, light→dark) — magnitude gets one hue, not
# a rainbow. Matches the palette's step-100→step-700 blue ramp.
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

CHART_SURFACE   = "#fcfcfb"
PRIMARY_INK     = "#0b0b0b"
SECONDARY_INK   = "#52514e"
MUTED_INK       = "#898781"
GRIDLINE        = "#e1e0d9"


def plot_confusion_matrix(confusion_rows, path):
    if not confusion_rows:
        print(f"No data for {path}")
        return

    categories_present = {r["manual_result"] for r in confusion_rows} | {r["pipeline_result"] for r in confusion_rows}
    ordered = [c for c in CATEGORY_ORDER if c in categories_present]
    extra = sorted(categories_present - set(ordered))
    categories = ordered + extra
    index = {c: i for i, c in enumerate(categories)}

    n = len(categories)
    matrix = [[0] * n for _ in range(n)]
    for r in confusion_rows:
        matrix[index[r["manual_result"]]][index[r["pipeline_result"]]] = r["count"]

    # Color by each row's own relative count, not the absolute count 
    # ("No fit" is the large majority, an absolute color scale renders every other row as the same color)
    row_totals = [sum(row) or 1 for row in matrix]
    row_share = [[matrix[i][j] / row_totals[i] for j in range(n)] for i in range(n)]

    cmap = LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL_BLUE)

    fig, ax = plt.subplots(figsize=(1.15 * n + 2.2, 1.15 * n + 1.8), facecolor=CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)
    im = ax.imshow(row_share, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_xticklabels(categories, rotation=30, ha="right", color=SECONDARY_INK)
    ax.set_yticks(range(n))
    ax.set_yticklabels(categories, color=SECONDARY_INK)
    ax.set_xlabel("Pipeline result", color=SECONDARY_INK)
    ax.set_ylabel("Manual category", color=SECONDARY_INK)
    ax.set_title("Manual vs. pipeline result — confusion matrix", color=PRIMARY_INK, fontsize=13, pad=14)

    ax.set_xticks([x - 0.5 for x in range(1, n)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, n)], minor=True)
    ax.grid(which="minor", color=GRIDLINE, linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", color=MUTED_INK, labelcolor=SECONDARY_INK)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(n):
        for j in range(n):
            count = matrix[i][j]
            if count == 0:
                continue
            share = row_share[i][j]
            text_color = "#ffffff" if share > 0.55 else PRIMARY_INK
            ax.text(j, i - 0.12, str(count), ha="center", va="center", color=text_color, fontsize=11)
            ax.text(j, i + 0.18, f"{share:.0%}", ha="center", va="center", color=text_color, fontsize=8, alpha=0.85)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    cbar.ax.set_ylabel("Share of row (manual category)", color=SECONDARY_INK)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    cbar.ax.yaxis.set_tick_params(color=MUTED_INK, labelcolor=SECONDARY_INK)

    fig.tight_layout()
    fig.savefig(path, facecolor=CHART_SURFACE, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  -> {path}")


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
    print(f"\n{'Hero Type':<40} {'Chars':>6} {'Compared':>9} {'Match':>6} {'Mismatch':>9} {'Match Rate':>10}")
    print("-" * 90)
    for r in summary:
        rate = r["match_rate"]
        rate_str = f"{rate:.3f}" if rate is not None else "N/A"
        print(f"{r['hero_type']:<40} {r['n_characters']:>6} {r['n_compared']:>9} "
              f"{r['n_match']:>6} {r['n_mismatch']:>9} {rate_str:>10}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare LLM hero type results against manual annotations by checking "
                     "whether the pipeline's categorical result ('result' column) matches "
                     "the manually assessed category, for every hero type, for each character "
                     f"that appears anywhere in the manual index. Hero types the manual index "
                     f"doesn't explicitly list for a character default to '{DEFAULT_MANUAL_RESULT}'. "
                     "Characters absent from the manual index entirely are ignored."
    )
    parser.add_argument(
        "--model", choices=["gemini", "qwen"], required=True,
        help="Which model's results to evaluate"
    )
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    results_csv = cfg["results_csv"]
    output_dir = cfg["output_dir"] / "comparison_manual"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:        {args.model}")
    print(f"Ground truth: manual ({GROUND_TRUTH_JSON})")
    print(f"Results CSV:  {results_csv}")
    print(f"Output dir:   {output_dir}")

    manual = load_manual_index(GROUND_TRUTH_JSON)
    hero_type_names, rows_by_key = load_results(results_csv)
    print(f"  {len(rows_by_key)} character rows, {len(hero_type_names)} hero types")

    characters = build_assessed_characters(manual, hero_type_names)
    print(f"  {len(characters)} manually-assessed characters "
          f"(unlisted hero types default to '{DEFAULT_MANUAL_RESULT}')")

    detail_rows = compare(characters, hero_type_names, rows_by_key)
    summary = summarise(hero_type_names, detail_rows)
    print_summary(summary)

    category_accuracy = accuracy_by_category(detail_rows)
    print_accuracy_by_category(category_accuracy)

    confusion_rows = confusion_matrix(detail_rows)

    write_csv(detail_rows, output_dir / "comparison_detail.csv")
    write_csv(summary, output_dir / "comparison_summary.csv")
    write_csv(confusion_rows, output_dir / "confusion_matrix.csv")
    write_csv(category_accuracy, output_dir / "comparison_by_category.csv")
    write_csv(character_summary(detail_rows), output_dir / "comparison_character_summary.csv")
    plot_confusion_matrix(confusion_rows, output_dir / "confusion_matrix.pdf")
    print("Done.")

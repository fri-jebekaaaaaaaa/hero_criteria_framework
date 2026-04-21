import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

FRAMEWORK_JSON  = Path("framework/Hero_Type_Criteria_Framework_v0.1.json")
ASSESSMENT_DIR  = Path("qwen_assessment_output/criteria_assessment_output")
OUTPUT_CSV      = Path("qwen_assessment_output/hero_type_results_qwen.csv")

#FRAMEWORK_JSON  = Path("framework/Hero_Type_Criteria_Framework_v0.1.json")
#ASSESSMENT_DIR  = Path("gemini_assessment_output/criteria_assessment_output")
#OUTPUT_CSV      = Path("gemini_assessment_output/hero_type_results_gemini.csv")


# Glob pattern to find all assessment files
ASSESSMENT_GLOB = "**/*__criteria_assessment.json"


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORICAL RESULT
# ─────────────────────────────────────────────────────────────────────────────

def categorical_result(
    matched_nec:   int,
    total_nec:     int,
    matched_sup:   int,
    total_sup:     int,
    matched_exc:   int,
) -> str:
    # 1. Disqualified — any exclusion matched
    if matched_exc > 0:
        return "Disqualified"

    nec_pct = matched_nec / total_nec if total_nec > 0 else 0.0
    sup_pct = matched_sup / total_sup if total_sup > 0 else 0.0

    # 2. No fit — nothing matched at all
    if matched_nec == 0 and matched_sup == 0:
        return "No fit"

    # 3. Definitive — all necessary criteria met (split by supporting richness)
    if nec_pct == 1.0:
        if sup_pct > 0.30:
            return "Definitive - strong support"
        else:
            return "Definitive - low support"

    # 4. Strong — >60% necessary (split by supporting richness)
    if nec_pct > 0.60:
        if sup_pct > 0.30:
            return "Strong - good support"
        else:
            return "Strong - low support"

    # 5. Partial — ≤60% necessary but >30% supporting
    if sup_pct > 0.30:
        return "Partial"

    # 6. Weak — low signal on both, but something matched
    return "Weak"


# ─────────────────────────────────────────────────────────────────────────────
# LOAD FRAMEWORK TOTALS
# ─────────────────────────────────────────────────────────────────────────────

def load_framework(path: Path) -> Dict[str, Dict[str, Any]]:
    """Return {hero_type_name: hero_type_dict}."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {ht["hero_type"]: ht for ht in data["hero_types"]}


# ─────────────────────────────────────────────────────────────────────────────
# PARSE ONE ASSESSMENT FILE
# ─────────────────────────────────────────────────────────────────────────────

def parse_assessment_file(
    path: Path,
    framework: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    # return a list of row-dicts, one per character in the file
    # each dict has:
    #    tale, character, book,
    #    one entry per hero type keyed as (hero_type, metric) 
    #       (metric = {"necessary", "supporting", "exclusion", "result"})

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    tale      = data.get("tale", "")
    character = data.get("character", "")
    # derive book name from the path (assessments live in <output_dir>/<book>/)
    book = path.parent.name

    row: Dict[str, Any] = {
        "book":      book,
        "tale":      tale,
        "character": character,
    }

    for assessment in data.get("assessments", []):
        hero_type_name = assessment.get("hero_type", "")

        if "error" in assessment:
            # step 1 failed for this pair — mark as missing
            row[(hero_type_name, "necessary")]  = ""
            row[(hero_type_name, "supporting")] = ""
            row[(hero_type_name, "exclusion")]  = ""
            row[(hero_type_name, "result")]     = "error"
            continue

        # ── necessary ────────────────────────────────────────────────────────
        necessary_items  = assessment.get("necessary", [])
        matched_nec      = sum(1 for n in necessary_items if n.get("value") is True)
        total_nec        = len(necessary_items)

        ht_def           = framework.get(hero_type_name, {})
        necessary_logic  = ht_def.get("necessary_logic", "AND").upper()

        # OR logic: any single match satisfies all necessary criteria → treat as 100%
        if necessary_logic == "OR" and matched_nec >= 1:
            effective_matched_nec = total_nec   # normalise to 100%
        else:
            effective_matched_nec = matched_nec

        # ── supporting ───────────────────────────────────────────────────────
        matched_sup = len(assessment.get("supporting_matched", []))
        total_sup   = len(ht_def.get("supporting_criteria", []))

        # ── exclusion ────────────────────────────────────────────────────────
        matched_exc = len(assessment.get("exclusion_matched", []))
        total_exc   = len(ht_def.get("exclusion_criteria", []))

        result = categorical_result(
            effective_matched_nec, total_nec,
            matched_sup, total_sup,
            matched_exc,
        )

        row[(hero_type_name, "necessary")]  = f"{matched_nec}/{total_nec}"
        row[(hero_type_name, "supporting")] = f"{matched_sup}/{total_sup}"
        row[(hero_type_name, "exclusion")]  = f"{matched_exc}/{total_exc}"
        row[(hero_type_name, "result")]     = result

    return row


# ─────────────────────────────────────────────────────────────────────────────
# BUILD CSV
# ─────────────────────────────────────────────────────────────────────────────

def build_csv(
    assessment_dir: Path,
    framework: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    assessment_files = sorted(assessment_dir.glob(ASSESSMENT_GLOB))
    print(f"Found {len(assessment_files)} assessment file(s)")

    # parse all files into rows
    rows: List[Dict[str, Any]] = []
    for path in assessment_files:
        row = parse_assessment_file(path, framework)
        rows.append(row)
        print(f"  Parsed {path.parent.name}/{path.name}")

    if not rows:
        print("No rows to write.")
        return

    # collect all hero type names in a stable order (framework order)
    hero_type_names: List[str] = list(framework.keys())

    # fixed header columns
    fixed_cols = ["book", "tale", "character"]

    # build full header: for each hero type, 4 sub-columns
    hero_cols: List[Tuple[str, str]] = []
    for ht_name in hero_type_names:
        for metric in ("necessary", "supporting", "exclusion", "result"):
            hero_cols.append((ht_name, metric))

    # write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # row 1: hero type names (merged across 4 columns, but CSV has no merge —
        #         we repeat the name for each of its 4 sub-columns)
        writer.writerow(
            fixed_cols
            + [ht_name for ht_name, _ in hero_cols]
        )

        # row 2: metric sub-headers
        writer.writerow(
            ["", "", ""]
            + [metric for _, metric in hero_cols]
        )

        # data rows
        for row in rows:
            data_row = [row.get("book", ""), row.get("tale", ""), row.get("character", "")]
            for key in hero_cols:
                data_row.append(row.get(key, ""))
            writer.writerow(data_row)

    print(f"\nCSV written → {output_path}")
    print(f"  {len(rows)} character rows  ×  {len(hero_type_names)} hero types")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loading framework from {FRAMEWORK_JSON}")
    framework = load_framework(FRAMEWORK_JSON)
    print(f"  {len(framework)} hero types loaded")

    build_csv(
        assessment_dir=ASSESSMENT_DIR,
        framework=framework,
        output_path=OUTPUT_CSV,
    )

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

FRAMEWORK_JSON = Path("framework/Hero_Type_Criteria_Framework_v0.1.json")
CSV_PATH       = Path("qwen_assessment_output/hero_type_results_qwen.csv")
OUTPUT_DIR     = Path("qwen_assessment_output/knowledge_graph")

#FRAMEWORK_JSON = Path("framework/Hero_Type_Criteria_Framework_v0.1.json")
#CSV_PATH       = Path("gemini_assessment_output/hero_type_results_gemini.csv")
#OUTPUT_DIR     = Path("gemini_assessment_output/knowledge_graph")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Only include FITS edges at or above this strength level
# 0 = all results except "No fit" and "Disqualified"
# 2 = Partial and above
# 3 = Strong and above
# 5 = Definitive only
MIN_RESULT_STRENGTH = 2

RESULT_ORDER: Dict[str, int] = {
    "Definitive - strong support": 6,
    "Definitive - low support":    5,
    "Strong - good support":       4,
    "Strong - low support":        3,
    "Partial":                     2,
    "Weak":                        1,
    "No fit":                      0,
    "Disqualified":               -1,
    "error":                      -2,
}

# node colours for Gephi (hex, no #)
NODE_COLORS = {
    "Book":      "4E79A7",   # blue
    "Tale":      "F28E2B",   # orange
    "Character": "E15759",   # red
    "HeroType":  "76B7B2",   # teal
    "Category":  "59A14F",   # green
}


# ─────────────────────────────────────────────────────────────────────────────
# LOAD FRAMEWORK
# ─────────────────────────────────────────────────────────────────────────────

def load_framework(path: Path) -> Dict[str, Dict[str, Any]]:
    """Return {hero_type_name: hero_type_dict}."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {ht["hero_type"]: ht for ht in data["hero_types"]}


# ─────────────────────────────────────────────────────────────────────────────
# PARSE CSV  (two-row header)
# ─────────────────────────────────────────────────────────────────────────────

def parse_csv(path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Returns (hero_type_names, rows).
    Each row is a dict with keys:
        book, tale, character,
        and {hero_type: {necessary, supporting, exclusion, result}} for each hero type.
    """
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header1 = next(reader)   # hero type names (repeated 4x each)
        header2 = next(reader)   # metric names

        # Build column index → (hero_type, metric)
        col_map: Dict[int, Tuple[str, str]] = {}
        hero_type_names: List[str] = []
        seen: Set[str] = set()

        for i, (ht, metric) in enumerate(zip(header1, header2)):
            if ht in ("book", "tale", "character") or ht == "":
                continue
            col_map[i] = (ht, metric)
            if ht not in seen:
                hero_type_names.append(ht)
                seen.add(ht)

        rows: List[Dict[str, Any]] = []
        for raw in reader:
            if not any(raw):
                continue
            row: Dict[str, Any] = {
                "book":      raw[0].strip(),
                "tale":      raw[1].strip(),
                "character": raw[2].strip(),
            }
            ht_data: Dict[str, Dict[str, str]] = defaultdict(dict)
            for i, (ht, metric) in col_map.items():
                if i < len(raw):
                    ht_data[ht][metric] = raw[i].strip()
            row["hero_types"] = dict(ht_data)
            rows.append(row)

    return hero_type_names, rows


# ─────────────────────────────────────────────────────────────────────────────
# NODE ID HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def book_id(book: str)      -> str: return f"book::{book}"
def tale_id(tale: str)      -> str: return f"tale::{tale}"
def char_id(tale: str, character: str) -> str: return f"char::{tale}::{character}"
def ht_id(hero_type: str)   -> str: return f"ht::{hero_type}"
def cat_id(category: str)   -> str: return f"cat::{category}"


# ─────────────────────────────────────────────────────────────────────────────
# BUILD GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(
    hero_type_names: List[str],
    rows: List[Dict[str, Any]],
    framework: Dict[str, Dict[str, Any]],
    min_strength: int = MIN_RESULT_STRENGTH,
) -> nx.DiGraph:

    G = nx.DiGraph()

    def add_node(node_id: str, node_type: str, label: str, **attrs) -> None:
        if node_id not in G:
            G.add_node(
                node_id,
                label=label,
                node_type=node_type,
                color=NODE_COLORS.get(node_type, "AAAAAA"),
                **attrs,
            )

    # ── HeroType + Category nodes (from framework) ───────────────────────────
    for ht_name, ht_def in framework.items():
        category = ht_def.get("category", "Unknown")
        definition = ht_def.get("definition", "")

        add_node(ht_id(ht_name), "HeroType", ht_name,
                 category=category, definition=definition)
        add_node(cat_id(category), "Category", category)

        G.add_edge(ht_id(ht_name), cat_id(category),
                   edge_type="BELONGS_TO")

    # ── Book / Tale / Character nodes + FITS edges ───────────────────────────
    for row in rows:
        book      = row["book"]
        tale      = row["tale"]
        character = row["character"]

        if not book or not tale or not character:
            continue

        b_id = book_id(book)
        t_id = tale_id(tale)
        c_id = char_id(tale, character)

        add_node(b_id, "Book", book.replace("_", " ").title())
        add_node(t_id, "Tale", tale, book=book)
        add_node(c_id, "Character", character, tale=tale, book=book)

        if not G.has_edge(b_id, t_id):
            G.add_edge(b_id, t_id, edge_type="CONTAINS")
        if not G.has_edge(t_id, c_id):
            G.add_edge(t_id, c_id, edge_type="FEATURES")

        for ht_name, metrics in row.get("hero_types", {}).items():
            result    = metrics.get("result", "")
            strength  = RESULT_ORDER.get(result, -99)

            if strength < min_strength:
                continue

            necessary  = metrics.get("necessary", "")
            supporting = metrics.get("supporting", "")
            exclusion  = metrics.get("exclusion", "")

            G.add_edge(
                c_id,
                ht_id(ht_name),
                edge_type="FITS",
                result=result,
                strength=strength,
                necessary=necessary,
                supporting=supporting,
                exclusion=exclusion,
            )

    return G


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(G: nx.DiGraph) -> None:
    node_counts = defaultdict(int)
    for _, data in G.nodes(data=True):
        node_counts[data.get("node_type", "?")] += 1

    edge_counts = defaultdict(int)
    for _, _, data in G.edges(data=True):
        edge_counts[data.get("edge_type", "?")] += 1

    print("\nGraph statistics:")
    print(f"  Total nodes : {G.number_of_nodes()}")
    for ntype, count in sorted(node_counts.items()):
        print(f"    {ntype:<15} {count}")
    print(f"  Total edges : {G.number_of_edges()}")
    for etype, count in sorted(edge_counts.items()):
        print(f"    {etype:<15} {count}")

    # Top characters by number of FITS edges
    fits_per_char = defaultdict(int)
    for src, dst, data in G.edges(data=True):
        if data.get("edge_type") == "FITS":
            label = G.nodes[src].get("label", src)
            fits_per_char[label] += 1

    print("\n  Top 10 characters by hero type fits:")
    for char, count in sorted(fits_per_char.items(), key=lambda x: -x[1])[:10]:
        print(f"    {char:<40} {count} fits")

    # Top hero types by number of characters that fit
    chars_per_ht = defaultdict(int)
    for src, dst, data in G.edges(data=True):
        if data.get("edge_type") == "FITS":
            label = G.nodes[dst].get("label", dst)
            chars_per_ht[label] += 1

    print("\n  Top 10 hero types by character count:")
    for ht, count in sorted(chars_per_ht.items(), key=lambda x: -x[1])[:10]:
        print(f"    {ht:<40} {count} characters")

    # Co-occurrence: which hero types appear together most often
    print("\n  Top 10 hero type co-occurrences (same character):")
    cooccur: Dict[Tuple[str, str], int] = defaultdict(int)
    char_to_hts: Dict[str, List[str]] = defaultdict(list)
    for src, dst, data in G.edges(data=True):
        if data.get("edge_type") == "FITS":
            char_to_hts[src].append(G.nodes[dst].get("label", dst))
    for char, hts in char_to_hts.items():
        hts_sorted = sorted(hts)
        for i in range(len(hts_sorted)):
            for j in range(i + 1, len(hts_sorted)):
                cooccur[(hts_sorted[i], hts_sorted[j])] += 1
    for pair, count in sorted(cooccur.items(), key=lambda x: -x[1])[:10]:
        print(f"    {pair[0]} + {pair[1]}: {count}")


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_graph(G: nx.DiGraph, output_dir: Path) -> None:
    # GEXF — best for Gephi (supports node/edge attributes + colour)
    gexf_path = output_dir / "hero_type_graph.gexf"
    nx.write_gexf(G, gexf_path)
    print(f"\n  GEXF  → {gexf_path}")

    # GraphML — general purpose, widely supported
    graphml_path = output_dir / "hero_type_graph.graphml"
    nx.write_graphml(G, graphml_path)
    print(f"  GraphML → {graphml_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loading framework: {FRAMEWORK_JSON}")
    framework = load_framework(FRAMEWORK_JSON)
    print(f"  {len(framework)} hero types")

    print(f"\nParsing CSV: {CSV_PATH}")
    hero_type_names, rows = parse_csv(CSV_PATH)
    print(f"  {len(rows)} character rows, {len(hero_type_names)} hero types")

    print(f"\nBuilding graph (min_strength={MIN_RESULT_STRENGTH}: "
          f"{[k for k,v in RESULT_ORDER.items() if v >= MIN_RESULT_STRENGTH]}) ...")
    G = build_graph(hero_type_names, rows, framework, MIN_RESULT_STRENGTH)

    print_stats(G)

    print("\nExporting ...")
    export_graph(G, OUTPUT_DIR)

    print("\nDone.")


import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from vllm import LLM, SamplingParams

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TALES_ROOT     = Path("folk_stories")
FRAMEWORK_JSON = Path("hero_type_criteria/Hero_Type_Criteria_Framework_v0.1.json")
OUTPUT_DIR     = Path("criteria_assessment_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHARACTERS_CONFIG = Path("folk_stories/characters_config.json")

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# token budgets per step
STEP1_MAX_TOKENS = 4096  # reasoning only, no quotes → shorter
STEP2_MAX_TOKENS = 8192  # quotes only, subset of criteria → shorter

# vLLM settings for H100 80 GB
GPU_MEMORY_UTILIZATION = 0.9   # leaves ~9 GB headroom for KV cache
MAX_MODEL_LEN          = 32768


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL — loaded once, used for both steps
# ─────────────────────────────────────────────────────────────────────────────

log(f"Loading model with vLLM: {MODEL_NAME}")
llm = LLM(
    model=MODEL_NAME,
    dtype="bfloat16",
    max_model_len=MAX_MODEL_LEN,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    enable_prefix_caching=True,   # share tale text across step 1 & 2 → cache hits
    trust_remote_code=True,
)
tokenizer = llm.get_tokenizer()
log("Model ready")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_framework(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    framework = data["hero_types"]
    log(f"Loaded framework: {len(framework)} hero types")
    return framework


def load_tale(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_characters_config(path: Path) -> Dict[str, Dict[str, List[str]]]:
    """Load characters_config.json → {book_name: {filename: [characters]}}."""
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    total = sum(
        sum(len(chars) for chars in stories.values())
        for stories in config.values()
    )
    log(f"Loaded characters config: {len(config)} book(s), {total} character entry/entries total")
    return config


def already_done(tale_stem: str, character: str, output_dir: Path) -> bool:
    """Return True if the output file for this (tale, character) already exists."""
    safe_tale = re.sub(r"[^a-zA-Z0-9_-]+", "_", tale_stem)
    safe_char = re.sub(r"[^a-zA-Z0-9_-]+", "_", character)
    return (output_dir / f"{safe_tale}__{safe_char}__criteria_assessment.json").exists()


# ─────────────────────────────────────────────────────────────────────────────
# JSON HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def extract_first_json_object(text: str) -> str:
    """Locate the first balanced { … } block in raw model output."""
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found:\n{text[:300]}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError(f"Unclosed JSON object:\n{text[:500]}")


def safe_parse(raw: str, label: str = "") -> Optional[Dict[str, Any]]:
    try:
        return json.loads(extract_first_json_object(raw))
    except Exception as e:
        log(f"JSON parse error [{label}]: {e} | raw[:200]={raw[:200]}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────────────────────────────────────

def render_chat(system: str, user: str) -> str:
    """Apply Qwen3 chat template with thinking disabled."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,      # suppress <think> tokens
    )
 

# STEP 1 — CRITERIA EVALUATION

_STEP1_SYSTEM = (
    "You are a literary analysis assistant. "
    "Evaluate narrative criteria for a specific character. "
    "Return only valid JSON. No markdown fences. No explanation outside JSON."
)
 
def build_step1_prompt(
    tale_title: str,
    tale_text: str,
    character: str,
    hero_type: Dict[str, Any],
) -> str:
    n_necessary = len(hero_type["necessary_criteria"])
 
    user = f"""
Assess character "{character}" in "{tale_title}" against the hero type below.
 
═══════════════════════════════
HERO TYPE
═══════════════════════════════
Name:       {hero_type["hero_type"]}
Category:   {hero_type["category"]}
Definition: {hero_type["definition"]}
 
NECESSARY CRITERIA
{json.dumps([{"index": i, "criterion": c} for i, c in enumerate(hero_type["necessary_criteria"])], ensure_ascii=False, indent=2)}
 
SUPPORTING CRITERIA
{json.dumps([{"index": i, "criterion": c} for i, c in enumerate(hero_type["supporting_criteria"])], ensure_ascii=False, indent=2)}
 
EXCLUSION CRITERIA
{json.dumps([{"index": i, "criterion": c} for i, c in enumerate(hero_type["exclusion_criteria"])], ensure_ascii=False, indent=2)}
 
═══════════════════════════════
FAIRY TALE
═══════════════════════════════
\"\"\"
{tale_text}
\"\"\"
 
═══════════════════════════════
INSTRUCTIONS
═══════════════════════════════
• Assess ONLY "{character}". Do NOT attribute other characters' actions or traits to them.
• For each NECESSARY criterion → true/false + 1-2 sentence reasoning.  No quotes here.
• For SUPPORTING → list ONLY clearly matched indexes + 1-2 sentence reasoning.
• For EXCLUSION → these are EDGE CASE FILTERS, not "not applicable" markers.
  Only match an exclusion criterion if ALL of the following are true:
    1. The character has at least some fit signal (necessary or supporting criteria partially matched)
    2. The exclusion criterion actively describes a limitation that overrides that fit
    3. You can point to specific evidence in the text
  If the character simply does not fit the hero type at all, leave exclusion_matched EMPTY.
  A character with 0 necessary and 0 supporting matches should NEVER have exclusion matches.
• Be conservative: if unsure, mark false / omit from matched lists.
 
═══════════════════════════════
OUTPUT FORMAT (strict JSON, no markdown)
═══════════════════════════════
{{
  "hero_type": "{hero_type["hero_type"]}",
  "necessary": [
    {{"index": 0, "value": false, "reasoning": "..."}}
  ],
  "supporting_matched": [
    {{"index": 0, "reasoning": "..."}}
  ],
  "exclusion_matched": [
    {{"index": 0, "reasoning": "..."}}
  ]
}}
 
"necessary" MUST contain EXACTLY {n_necessary} items (one per criterion, in order).
""".strip()
 
    return render_chat(_STEP1_SYSTEM, user)
 
 
def validate_step1(data: Dict[str, Any], hero_type: Dict[str, Any]) -> None:
    required = {"hero_type", "necessary", "supporting_matched", "exclusion_matched"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Step 1 missing keys: {missing}")
 
    expected = len(hero_type["necessary_criteria"])
    got = len(data["necessary"])
    if got != expected:
        raise ValueError(f"Step 1 necessary count: expected {expected}, got {got}")
 
    for item in data["necessary"]:
        if not isinstance(item.get("value"), bool):
            raise ValueError(f"Step 1 item missing boolean value: {item}")
 
 
# STEP 2 — QUOTE EXTRACTION

_STEP2_SYSTEM = (
    "You are a precise quote extraction assistant. "
    "Copy verbatim phrases from a fairy tale that directly support given criteria. "
    "Return only valid JSON. No markdown fences. No explanation outside JSON."
)
 
 
@dataclass
class QuoteTask:
    tale_title:        str
    tale_text:         str
    character:         str
    hero_type_name:    str
    necessary_true:    List[Dict[str, Any]] = field(default_factory=list)
    supporting_matched: List[Dict[str, Any]] = field(default_factory=list)
    exclusion_matched:  List[Dict[str, Any]] = field(default_factory=list)
 
    def is_empty(self) -> bool:
        return not (self.necessary_true or self.supporting_matched or self.exclusion_matched)
 
 
def build_step2_prompt(task: QuoteTask) -> str:
    sections: List[str] = []
    if task.necessary_true:
        sections.append(
            "NECESSARY (true — find supporting quotes):\n"
            + json.dumps(task.necessary_true, ensure_ascii=False, indent=2)
        )
    if task.supporting_matched:
        sections.append(
            "SUPPORTING (matched — find supporting quotes):\n"
            + json.dumps(task.supporting_matched, ensure_ascii=False, indent=2)
        )
    if task.exclusion_matched:
        sections.append(
            "EXCLUSION (matched — find supporting quotes):\n"
            + json.dumps(task.exclusion_matched, ensure_ascii=False, indent=2)
        )
 
    criteria_block = "\n\n".join(sections)
 
    user = f"""
Extract verbatim quotes from the fairy tale that support each criterion below.
 
Character: "{task.character}"
Story:     "{task.tale_title}"
 
═══════════════════════════════
CRITERIA TO SUPPORT
═══════════════════════════════
{criteria_block}
 
═══════════════════════════════
FAIRY TALE
═══════════════════════════════
\"\"\"
{task.tale_text}
\"\"\"
 
═══════════════════════════════
EVIDENCE RULES (CRITICAL)
═══════════════════════════════
• Every quote MUST be a VERBATIM copy — exact wording, no changes.
• The quote must directly refer to "{task.character}" (their actions, state, or what is said about them).
• Do NOT paraphrase. Do NOT summarize. Do NOT invent.
• Keep quotes short (one sentence or clause).
• MAX 3 quotes per criterion. Pick the most direct ones.
• If no direct quote exists → use [].
 
═══════════════════════════════
OUTPUT FORMAT (strict JSON)
═══════════════════════════════
{{
  "necessary": [
    {{"index": 0, "quotes": ["exact quote..."]}}
  ],
  "supporting_matched": [
    {{"index": 0, "quotes": ["exact quote..."]}}
  ],
  "exclusion_matched": [
    {{"index": 0, "quotes": ["exact quote..."]}}
  ]
}}
 
Include ONLY the indexes given to you above. Omit any section that has no entries.
""".strip()
 
    return render_chat(_STEP2_SYSTEM, user)
 
 
def validate_step2(data: Dict[str, Any]) -> None:
    for section in ("necessary", "supporting_matched", "exclusion_matched"):
        for item in data.get(section, []):
            if "index" not in item or "quotes" not in item:
                raise ValueError(f"Step 2 {section} item missing keys: {item}")
            if not isinstance(item["quotes"], list):
                raise ValueError(f"Step 2 quotes must be a list: {item}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MERGE STEP 1 & 2
# ─────────────────────────────────────────────────────────────────────────────
 
def merge_steps(
    step1: Dict[str, Any],
    step2: Optional[Dict[str, Any]],
    hero_type: Dict[str, Any],
) -> Dict[str, Any]:
     
    # build quote look-ups (index → list[str]) from Step 2
    s2_necessary:  Dict[int, List[str]] = {}
    s2_supporting: Dict[int, List[str]] = {}
    s2_exclusion:  Dict[int, List[str]] = {}
    if step2:
        for item in step2.get("necessary", []):
            s2_necessary[item["index"]] = item.get("quotes", [])
        for item in step2.get("supporting_matched", []):
            s2_supporting[item["index"]] = item.get("quotes", [])
        for item in step2.get("exclusion_matched", []):
            s2_exclusion[item["index"]] = item.get("quotes", [])
 
    necessary_out = [
        {
            "criterion": hero_type["necessary_criteria"][item["index"]]
                         if item["index"] < len(hero_type["necessary_criteria"])
                         else f"index_{item['index']}",
            "value":     item.get("value", False),
            "evidence":  s2_necessary.get(item["index"], []),
        }
        for item in step1.get("necessary", [])
    ]
 
    supporting_out = [
        {
            "criterion": hero_type["supporting_criteria"][item["index"]]
                         if item["index"] < len(hero_type["supporting_criteria"])
                         else f"index_{item['index']}",
            "evidence":  s2_supporting.get(item["index"], []),
        }
        for item in step1.get("supporting_matched", [])
    ]
 
    exclusion_out = [
        {
            "criterion": hero_type["exclusion_criteria"][item["index"]]
                         if item["index"] < len(hero_type["exclusion_criteria"])
                         else f"index_{item['index']}",
            "evidence":  s2_exclusion.get(item["index"], []),
        }
        for item in step1.get("exclusion_matched", [])
    ]
 
    # category and definition always come from the framework dict directly
    # so they are present even when step 1 output omitted them
    return {
        "hero_type":          hero_type["hero_type"],
        "category":           hero_type["category"],
        "definition":         hero_type["definition"],
        "necessary":          necessary_out,
        "supporting_matched": supporting_out,
        "exclusion_matched":  exclusion_out,
    }


# ─────────────────────────────────────────────────────────────────────────────
# JOB DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Job:
    tale_file:  Path
    character:  str
    hero_type:  Dict[str, Any]
    tale_text:  str

    @property
    def key(self) -> Tuple[str, str]:
        return (self.tale_file.stem, self.character)

    @property
    def label(self) -> str:
        return f"{self.tale_file.stem} | {self.character} | {self.hero_type['hero_type']}"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def build_story_jobs(
    tale_file: Path,
    characters: List[str],
    framework: List[Dict[str, Any]],
    output_dir: Path,
) -> List[Job]:
    # build all (character, hero_type) jobs for a single tale file.
    # characters whose output file already exists are skipped (checkpoint).
    
    pending_characters = [
        c for c in characters
        if not already_done(tale_file.stem, c, output_dir)
    ]

    if not pending_characters:
        return []

    tale_text = load_tale(tale_file)
    jobs: List[Job] = [
        Job(tale_file, character, hero_type, tale_text)
        for character in pending_characters
        for hero_type in framework
    ]
    return jobs


def run_step1(jobs: List[Job]) -> List[Optional[Dict[str, Any]]]:
    # build all Step 1 prompts and send them in a single vLLM batch.
    # returns a list aligned with `jobs`; failed parses are None.
    log("Building Step 1 prompts …")
    prompts = [
        build_step1_prompt(j.tale_file.stem, j.tale_text, j.character, j.hero_type)
        for j in jobs
    ]

    log(f"Running Step 1 — {len(prompts)} prompts …")
    t0 = time.time()

    outputs = llm.generate(
        prompts,
        [SamplingParams(temperature=0.0, max_tokens=STEP1_MAX_TOKENS) for job in jobs],
    )
    
    log(f"Step 1 done in {time.time() - t0:.1f}s")

    results: List[Optional[Dict[str, Any]]] = []
    for job, out in zip(jobs, outputs):
        raw    = out.outputs[0].text.strip()
        parsed = safe_parse(raw, label=f"S1 {job.label}")
        if parsed is None:
            results.append(None)
            continue
        try:
            validate_step1(parsed, job.hero_type)
            results.append(parsed)
        except ValueError as e:
            log(f"Step 1 validation fail | {job.label} | {e}")
            results.append(None)

    failed = sum(1 for r in results if r is None)
    log(f"Step 1 results: {len(results) - failed}/{len(results)} OK")
    return results


def run_step2(
    jobs: List[Job],
    step1_results: List[Optional[Dict[str, Any]]],
) -> Dict[int, Optional[Dict[str, Any]]]:
    # build Step 2 prompts only for jobs with at least one matched criterion,
    # send in a single vLLM batch, return a dict keyed by job index.
    tasks: List[Tuple[int, QuoteTask]] = []

    for job_idx, (job, s1) in enumerate(zip(jobs, step1_results)):
        if s1 is None:
            continue

        necessary_true = [
            {
                "index":     item["index"],
                "criterion": job.hero_type["necessary_criteria"][item["index"]],
                "reasoning": item.get("reasoning", ""),
            }
            for item in s1.get("necessary", [])
            if item.get("value") is True
            and item["index"] < len(job.hero_type["necessary_criteria"])
        ]

        supporting_matched = [
            {
                "index":     item["index"],
                "criterion": job.hero_type["supporting_criteria"][item["index"]],
                "reasoning": item.get("reasoning", ""),
            }
            for item in s1.get("supporting_matched", [])
            if item["index"] < len(job.hero_type["supporting_criteria"])
        ]

        exclusion_matched = [
            {
                "index":     item["index"],
                "criterion": job.hero_type["exclusion_criteria"][item["index"]],
                "reasoning": item.get("reasoning", ""),
            }
            for item in s1.get("exclusion_matched", [])
            if item["index"] < len(job.hero_type["exclusion_criteria"])
        ]

        task = QuoteTask(
            tale_title=job.tale_file.stem,
            tale_text=job.tale_text,
            character=job.character,
            hero_type_name=job.hero_type["hero_type"],
            necessary_true=necessary_true,
            supporting_matched=supporting_matched,
            exclusion_matched=exclusion_matched,
        )
        if not task.is_empty():
            tasks.append((job_idx, task))

    log(f"Step 2 tasks: {len(tasks)} (skipped {len(jobs) - len(tasks)} with no matches)")

    result_map: Dict[int, Optional[Dict[str, Any]]] = {}
    if not tasks:
        return result_map

    prompts = [build_step2_prompt(task) for _, task in tasks]

    log(f"Running Step 2 — {len(prompts)} prompts …")
    t0 = time.time()

    outputs = llm.generate(
        prompts,
        [SamplingParams(temperature=0.0, max_tokens=STEP2_MAX_TOKENS) for _, task in tasks],
    )
    log(f"Step 2 done in {time.time() - t0:.1f}s")

    for (job_idx, task), out in zip(tasks, outputs):
        raw    = out.outputs[0].text.strip()
        parsed = safe_parse(raw, label=f"S2 {task.hero_type_name}")
        if parsed is None:
            result_map[job_idx] = None
            continue
        try:
            validate_step2(parsed)
            result_map[job_idx] = parsed
        except ValueError as e:
            log(f"Step 2 validation fail | {task.hero_type_name} | {e}")
            result_map[job_idx] = None

    return result_map


def save_results(
    jobs: List[Job],
    step1_results: List[Optional[Dict[str, Any]]],
    step2_map: Dict[int, Optional[Dict[str, Any]]],
    output_dir: Path,
) -> None:
    #merge per-job results, group by (tale, character), and write JSON files.
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for job_idx, (job, s1) in enumerate(zip(jobs, step1_results)):
        if s1 is None:
            assessment: Dict[str, Any] = {
                "hero_type":  job.hero_type.get("hero_type", "<unknown>"),
                "category":   job.hero_type.get("category", ""),
                "definition": job.hero_type.get("definition", ""),
                "error":      "step1_failed",
            }
        else:
            s2 = step2_map.get(job_idx)
            assessment = merge_steps(s1, s2, job.hero_type)

        grouped[job.key].append(assessment)

    for (tale_title, character), assessments in grouped.items():
        payload = {
            "tale":        tale_title,
            "character":   character,
            "assessments": assessments,
        }
        safe_tale = re.sub(r"[^a-zA-Z0-9_-]+", "_", tale_title)
        safe_char = re.sub(r"[^a-zA-Z0-9_-]+", "_", character)
        out_path  = output_dir / f"{safe_tale}__{safe_char}__criteria_assessment.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        log(f"Saved → {out_path}")



def run_collection() -> None:
    framework = load_framework(FRAMEWORK_JSON)
    config    = load_characters_config(CHARACTERS_CONFIG)

    book_names  = list(config.keys())
    story_count = sum(len(s) for s in config.values())
    log(f"Books: {len(book_names)} | Stories configured: {story_count}")

    for book_name, stories in config.items():
        book_tales_dir = TALES_ROOT / book_name
        book_output_dir = OUTPUT_DIR / book_name
        book_output_dir.mkdir(parents=True, exist_ok=True)

        log(f"── Book: {book_name} ({len(stories)} stories) ──")

        for filename, characters in sorted(stories.items()):
            tale_file = book_tales_dir / filename

            if not tale_file.exists():
                log(f"  MISSING {tale_file} — skipping")
                continue

            # skip entirely if every character in this story is already done
            pending = [c for c in characters if not already_done(tale_file.stem, c, book_output_dir)]
            if not pending:
                log(f"  SKIP {filename} — all {len(characters)} character(s) already processed")
                continue

            log(f"  Processing {filename} | pending characters: {pending}")

            jobs = build_story_jobs(tale_file, characters, framework, book_output_dir)
            if not jobs:
                continue

            log(f"    Jobs: {len(jobs)} ({len(pending)} character(s) × {len(framework)} hero types)")

            step1_results = run_step1(jobs)
            step2_map     = run_step2(jobs, step1_results)
            save_results(jobs, step1_results, step2_map, book_output_dir)

    log("All done.")


if __name__ == '__main__':
    try:
        run_collection()
    finally:
        try:
            del llm
        except Exception:
            pass
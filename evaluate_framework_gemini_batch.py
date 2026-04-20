import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TALES_ROOT        = Path("folk_stories")
FRAMEWORK_JSON    = Path("hero_type_criteria/Hero_Type_Criteria_Framework_v0.1.json")
CHARACTERS_CONFIG = Path("folk_stories/characters_config.json")
OUTPUT_DIR        = Path("gemini_assessment_output/criteria_assessment_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# temporary JSONL files for batch input
WORK_DIR = Path("gemini_batch_work")
WORK_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "gemini-3-flash-preview"

# poll interval in seconds
POLL_INTERVAL = 30


client = genai.Client()

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
 
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
 

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_framework(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    framework = data["hero_types"]
    log(f"Loaded framework: {len(framework)} hero types")
    return framework
 
 
def load_tale(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()
 
 
def load_characters_config(path: Path) -> Dict[str, Dict[str, List[str]]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
 
 
def already_done(tale_stem: str, character: str, output_dir: Path) -> bool:
    safe_tale = re.sub(r"[^a-zA-Z0-9_-]+", "_", tale_stem)
    safe_char = re.sub(r"[^a-zA-Z0-9_-]+", "_", character)
    return (output_dir / f"{safe_tale}__{safe_char}__criteria_assessment.json").exists()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# JSON HELPERS
# ─────────────────────────────────────────────────────────────────────────────
 
def sanitise_json_string(raw: str) -> str:
    raw = raw.replace("\u201c", "'").replace("\u201d", "'")
    raw = raw.replace("\u2018", "'").replace("\u2019", "'")
    raw = "".join(ch for ch in raw if ch >= " " or ch in "\t\n\r")
    return raw
 
 
def extract_first_json_object(text: str) -> str:
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
                return text[start:i + 1]
    raise ValueError(f"Unclosed JSON object:\n{text[:500]}")
 
 
def safe_parse(raw: str, label: str = "") -> Optional[Dict[str, Any]]:
    try:
        return json.loads(extract_first_json_object(sanitise_json_string(raw)))
    except Exception as e:
        log(f"JSON parse error [{label}]: {e} | raw[:200]={raw[:200]}")
        return None
 
 
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
 
    @property
    def request_key(self) -> str:
        """Unique key for JSONL request identification."""
        safe_tale = re.sub(r"[^a-zA-Z0-9_-]+", "_", self.tale_file.stem)
        safe_char = re.sub(r"[^a-zA-Z0-9_-]+", "_", self.character)
        safe_ht   = re.sub(r"[^a-zA-Z0-9_-]+", "_", self.hero_type["hero_type"])
        return f"{safe_tale}__{safe_char}__{safe_ht}"
 
 
@dataclass
class QuoteTask:
    job_idx:           int
    tale_title:        str
    tale_text:         str
    character:         str
    hero_type_name:    str
    necessary_true:    List[Dict[str, Any]] = field(default_factory=list)
    supporting_matched: List[Dict[str, Any]] = field(default_factory=list)
    exclusion_matched:  List[Dict[str, Any]] = field(default_factory=list)
 
    def is_empty(self) -> bool:
        return not (self.necessary_true or self.supporting_matched or self.exclusion_matched)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────────────────────────────────────
 
_STEP1_SYSTEM = (
    "You are a literary analysis assistant. "
    "Evaluate narrative criteria for a specific character. "
    "Return only valid JSON. No markdown fences. No explanation outside JSON."
)
 
_STEP2_SYSTEM = (
    "You are a precise quote extraction assistant. "
    "Copy verbatim phrases from a fairy tale that directly support given criteria. "
    "Return only valid JSON. No markdown fences. No explanation outside JSON."
)
 
 
def build_step1_contents(job: Job) -> List[Dict]:
    n_necessary = len(job.hero_type["necessary_criteria"])
    user = f"""
Assess character "{job.character}" in "{job.tale_file.stem}" against the hero type below.
 
═══════════════════════════════
HERO TYPE
═══════════════════════════════
Name:       {job.hero_type["hero_type"]}
Category:   {job.hero_type["category"]}
Definition: {job.hero_type["definition"]}
 
NECESSARY CRITERIA
{json.dumps([{"index": i, "criterion": c} for i, c in enumerate(job.hero_type["necessary_criteria"])], ensure_ascii=False, indent=2)}
 
SUPPORTING CRITERIA
{json.dumps([{"index": i, "criterion": c} for i, c in enumerate(job.hero_type["supporting_criteria"])], ensure_ascii=False, indent=2)}
 
EXCLUSION CRITERIA
{json.dumps([{"index": i, "criterion": c} for i, c in enumerate(job.hero_type["exclusion_criteria"])], ensure_ascii=False, indent=2)}
 
═══════════════════════════════
FAIRY TALE
═══════════════════════════════
\"\"\"
{job.tale_text}
\"\"\"
 
═══════════════════════════════
INSTRUCTIONS
═══════════════════════════════
• Assess ONLY "{job.character}". Do NOT attribute other characters' actions or traits to them.
• For each NECESSARY criterion → true/false + 1 sentence reasoning (2 sentences maximum). No quotes here.
• For SUPPORTING → list ONLY indexes that clearly apply. 1 sentence reasoning each.
• For EXCLUSION → these are EDGE CASE FILTERS, not "not applicable" markers.
  Only match an exclusion criterion if ALL of the following are true:
    1. The character has at least some fit signal (necessary or supporting criteria partially matched)
    2. The exclusion criterion actively describes a limitation that overrides that fit
    3. You can point to specific evidence in the text
  If the character simply does not fit the hero type at all, leave exclusion_matched EMPTY.
  A character with 0 necessary and 0 supporting matches should NEVER have exclusion matches.
• Be conservative: if unsure, mark necessary false / omit from supporting / omit from exclusion.
 
═══════════════════════════════
OUTPUT FORMAT (strict JSON, no markdown)
═══════════════════════════════
{{
  "hero_type": "{job.hero_type["hero_type"]}",
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
 
    return [
        {"role": "user", "parts": [
            {"text": f"[SYSTEM] {_STEP1_SYSTEM}\n\n{user}"}
        ]}
    ]
 
 
def build_step2_contents(task: QuoteTask) -> List[Dict]:
    sections = []
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
 
    return [
        {"role": "user", "parts": [
            {"text": f"[SYSTEM] {_STEP2_SYSTEM}\n\n{user}"}
        ]}
    ]
 
 
# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
 
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
 
 
def validate_step2(data: Dict[str, Any]) -> None:
    for section in ("necessary", "supporting_matched", "exclusion_matched"):
        for item in data.get(section, []):
            if "index" not in item or "quotes" not in item:
                raise ValueError(f"Step 2 {section} item missing keys: {item}")
            if not isinstance(item["quotes"], list):
                raise ValueError(f"Step 2 quotes must be a list: {item}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# BATCH HELPERS
# ─────────────────────────────────────────────────────────────────────────────
 
def write_jsonl(requests: List[Dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
    log(f"Wrote {len(requests)} requests → {path}")
 
 
def upload_jsonl(path: Path) -> str:
    """Upload a JSONL file via File API and return the file name."""
    log(f"Uploading {path.name} …")
    uploaded = client.files.upload(
        file=str(path),
        config=types.UploadFileConfig(
            display_name=path.stem,
            mime_type="jsonl",
        ),
    )
    log(f"Uploaded: {uploaded.name}")
    return uploaded.name
 
 
def submit_batch(file_name: str, display_name: str) -> Any:
    """Submit a batch job from an uploaded JSONL file."""
    log(f"Submitting batch job: {display_name} …")
    job = client.batches.create(
        model=MODEL_NAME,
        src=file_name,
        config={"display_name": display_name},
    )
    log(f"Batch job created: {job.name}")
    return job
 
 
def poll_batch(job_name: str) -> Any:
    """Poll until the batch job reaches a terminal state."""
    terminal = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
                "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
    log(f"Polling {job_name} …")
    while True:
        job = client.batches.get(name=job_name)
        state = job.state.name if hasattr(job.state, "name") else str(job.state)
        if state in terminal:
            log(f"Batch finished: {state}")
            return job
        log(f"  State: {state} — waiting {POLL_INTERVAL}s …")
        time.sleep(POLL_INTERVAL)
 
 
def download_results(job: Any, save_path: Optional[Path] = None) -> str:
    """Download the JSONL result file, save to disk, and return its text content."""
    result_file_name = job.dest.file_name
    log(f"Downloading results from {result_file_name} …")
    content_bytes = client.files.download(file=result_file_name)
    content = content_bytes.decode("utf-8")
    if save_path:
        save_path.write_text(content, encoding="utf-8")
        log(f"Saved raw results → {save_path}")
    return content
 
 
def parse_batch_results(content: str) -> Dict[str, str]:
    """
    Parse a JSONL result file into {request_key: raw_text}.
    Each line is a GenerateContentResponse or an error status object.
    """
    results: Dict[str, str] = {}
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            log(f"Could not parse result line: {line[:100]}")
            continue
 
        key = obj.get("key", "")
        if not key:
            continue
 
        # Success path
        response = obj.get("response")
        if response:
            try:
                text = response["candidates"][0]["content"]["parts"][0]["text"]
                results[key] = text
            except (KeyError, IndexError):
                log(f"Could not extract text for key={key}")
        else:
            error = obj.get("error", {})
            log(f"Error for key={key}: {error}")
 
    return results
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MERGE
# ─────────────────────────────────────────────────────────────────────────────
 
def merge_steps(
    step1: Dict[str, Any],
    step2: Optional[Dict[str, Any]],
    hero_type: Dict[str, Any],
) -> Dict[str, Any]:
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
            "reasoning": item.get("reasoning", ""),
            "evidence":  s2_necessary.get(item["index"], []),
        }
        for item in step1.get("necessary", [])
    ]
 
    supporting_out = [
        {
            "criterion": hero_type["supporting_criteria"][item["index"]]
                         if item["index"] < len(hero_type["supporting_criteria"])
                         else f"index_{item['index']}",
            "reasoning": item.get("reasoning", ""),
            "evidence":  s2_supporting.get(item["index"], []),
        }
        for item in step1.get("supporting_matched", [])
    ]
 
    exclusion_out = [
        {
            "criterion": hero_type["exclusion_criteria"][item["index"]]
                         if item["index"] < len(hero_type["exclusion_criteria"])
                         else f"index_{item['index']}",
            "reasoning": item.get("reasoning", ""),
            "evidence":  s2_exclusion.get(item["index"], []),
        }
        for item in step1.get("exclusion_matched", [])
    ]
 
    return {
        "hero_type":          hero_type["hero_type"],
        "category":           hero_type["category"],
        "definition":         hero_type["definition"],
        "necessary":          necessary_out,
        "supporting_matched": supporting_out,
        "exclusion_matched":  exclusion_out,
    }
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
 
def save_results(
    jobs: List[Job],
    step1_parsed: List[Optional[Dict[str, Any]]],
    step2_map: Dict[int, Optional[Dict[str, Any]]],
    output_dir: Path,
) -> None:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
 
    for job_idx, (job, s1) in enumerate(zip(jobs, step1_parsed)):
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
        payload = {"tale": tale_title, "character": character, "assessments": assessments}
        safe_tale = re.sub(r"[^a-zA-Z0-9_-]+", "_", tale_title)
        safe_char = re.sub(r"[^a-zA-Z0-9_-]+", "_", character)
        out_path  = output_dir / f"{safe_tale}__{safe_char}__criteria_assessment.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        log(f"Saved → {out_path}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
 
def build_jobs(
    framework: List[Dict[str, Any]],
    config: Dict[str, Dict[str, List[str]]],
) -> List[Job]:
    # build jobs for validated characters only, skipping already-done ones.
    jobs: List[Job] = []
    for book_name, stories in config.items():
        book_tales_dir  = TALES_ROOT / book_name
        book_output_dir = OUTPUT_DIR / book_name
        book_output_dir.mkdir(parents=True, exist_ok=True)
 
        for filename, characters in sorted(stories.items()):
            tale_file = book_tales_dir / filename
            if not tale_file.exists():
                continue
 
            validated = [
                c for c in characters
                if not already_done(tale_file.stem, c, book_output_dir)
            ]
            if not validated:
                continue
 
            tale_text = load_tale(tale_file)
            for character in validated:
                for hero_type in framework:
                    jobs.append(Job(tale_file, character, hero_type, tale_text))
 
    log(f"Total jobs: {len(jobs)}")
    return jobs
 
 
def run_step1_batch(jobs: List[Job]) -> List[Optional[Dict[str, Any]]]:
    # submit all Step 1 jobs as a single Gemini batch and return parsed results.
    log(f"Building {len(jobs)} Step 1 requests …")
 
    requests = []
    for job in jobs:
        requests.append({
            "key":     job.request_key,
            "request": {
                "contents":           build_step1_contents(job),
                "generation_config":  {"temperature": 0.0, "max_output_tokens": 65536},
            },
        })
 
    jsonl_path = WORK_DIR / "step1_requests.jsonl"
    write_jsonl(requests, jsonl_path)
 
    file_name  = upload_jsonl(jsonl_path)
    batch_job  = submit_batch(file_name, "hero-criteria-step1")
    batch_job  = poll_batch(batch_job.name)
 
    if (batch_job.state.name if hasattr(batch_job.state, "name") else str(batch_job.state)) != "JOB_STATE_SUCCEEDED":
        log("Step 1 batch did not succeed — aborting")
        return [None] * len(jobs)
 
    raw_results = parse_batch_results(download_results(batch_job, WORK_DIR / "step1_results.jsonl"))
 
    # map back by request_key
    key_to_idx = {job.request_key: i for i, job in enumerate(jobs)}
    parsed: List[Optional[Dict[str, Any]]] = [None] * len(jobs)
 
    for key, text in raw_results.items():
        idx = key_to_idx.get(key)
        if idx is None:
            log(f"Unknown key in Step 1 results: {key}")
            continue
        job  = jobs[idx]
        data = safe_parse(text, label=f"S1 {job.label}")
        if data is None:
            continue
        try:
            validate_step1(data, job.hero_type)
            parsed[idx] = data
        except ValueError as e:
            log(f"Step 1 validation fail | {job.label} | {e}")
 
    ok = sum(1 for p in parsed if p is not None)
    log(f"Step 1: {ok}/{len(jobs)} OK")
    return parsed
 
 
def run_step2_batch(
    jobs: List[Job],
    step1_results: List[Optional[Dict[str, Any]]],
) -> Dict[int, Optional[Dict[str, Any]]]:
    # build Step 2 tasks from Step 1 matches and submit as a Gemini batch.
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
            job_idx=job_idx,
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
 
    # build request key for step 2 (append __s2 to avoid collision with step 1)
    def s2_key(task: QuoteTask) -> str:
        base = jobs[task.job_idx].request_key
        return base + "__s2"
 
    requests = []
    for job_idx, task in tasks:
        requests.append({
            "key":     s2_key(task),
            "request": {
                "contents":           build_step2_contents(task),
                "generation_config":  {"temperature": 0.0, "max_output_tokens": 65536},
            },
        })
 
    jsonl_path = WORK_DIR / "step2_requests.jsonl"
    write_jsonl(requests, jsonl_path)
 
    file_name = upload_jsonl(jsonl_path)
    batch_job = submit_batch(file_name, "hero-criteria-step2")
    batch_job = poll_batch(batch_job.name)
 
    if (batch_job.state.name if hasattr(batch_job.state, "name") else str(batch_job.state)) != "JOB_STATE_SUCCEEDED":
        log("Step 2 batch did not succeed — returning empty map")
        return result_map
 
    raw_results = parse_batch_results(download_results(batch_job, WORK_DIR / "step2_results.jsonl"))
 
    # map back by s2_key
    key_to_task = {s2_key(task): (job_idx, task) for job_idx, task in tasks}
 
    for key, text in raw_results.items():
        entry = key_to_task.get(key)
        if entry is None:
            log(f"Unknown key in Step 2 results: {key}")
            continue
        job_idx, task = entry
        data = safe_parse(text, label=f"S2 {task.hero_type_name}")
        if data is None:
            result_map[job_idx] = None
            continue
        try:
            validate_step2(data)
            result_map[job_idx] = data
        except ValueError as e:
            log(f"Step 2 validation fail | {task.hero_type_name} | {e}")
            result_map[job_idx] = None
 
    return result_map
 
 
def run_collection() -> None:
    framework = load_framework(FRAMEWORK_JSON)
    config    = load_characters_config(CHARACTERS_CONFIG)
 
    jobs = build_jobs(framework, config)
    if not jobs:
        log("No jobs to process — all characters already done.")
        return
 
    log(f"Processing {len(jobs)} jobs "
        f"({len({j.key for j in jobs})} character×tale pairs × {len(framework)} hero types)")
 
    step1_results = run_step1_batch(jobs)
    step2_map     = run_step2_batch(jobs, step1_results)
 
    # save grouped by book
    book_jobs: Dict[str, List[Tuple[int, Job]]] = defaultdict(list)
    for idx, job in enumerate(jobs):
        book_jobs[job.tale_file.parent.name].append((idx, job))
 
    for book_name, indexed_jobs in book_jobs.items():
        book_output_dir = OUTPUT_DIR / book_name
        book_output_dir.mkdir(parents=True, exist_ok=True)
        idxs      = [i for i, _ in indexed_jobs]
        book_job_list  = [j for _, j in indexed_jobs]
        book_s1   = [step1_results[i] for i in idxs]
        book_s2   = {new_i: step2_map.get(old_i)
                     for new_i, old_i in enumerate(idxs) if old_i in step2_map}
        save_results(book_job_list, book_s1, book_s2, book_output_dir)
 
    log("All done.")
 
 
if __name__ == "__main__":
    run_collection()
# Hero Type Criteria Framework
Computational framework for identifying character archetypes in fiction using LLMs, based on Seal and White's Encyclopedia of Folk Heroes and Heroines Around the World (2016).

## Overview
The framework classifies characters from narrative text into 51 hero types using a two-step LLM pipeline:

1. Criteria Evaluation — the LLM evaluates a character against necessary, supporting, and exclusion criteria for each hero type.
2. Evidence Extraction — the LLM extracts verbatim quotes from the story to support each matched criterion.

Results are aggregated into categorical fit scores (Strong, Partial, Weak, No fit, Disqualified) and displayed as a knowledge graph.

## Repository Structure
```
├── folk_stories/ 
│   ├── andrew_lang_fairy_books/ # 12 books, 438 stories
│   ├── ... # additional collections (Grimm, Anansi, king Arthur, etc.)
│   ├── character_config.json # List of characters to analyse
│   └── seal_index_of_character_types.json  # Ground truth classification from Seal and White's encyclopedia
├── framework/
│   ├── Hero_Type_Annotation_Sheet.xlsx # Template spreadsheet for manual annotations
│   └── Hero_Type_Criteria_Framework.json  # 51 hero type definitions
├── manual_annotations/ # Results from manual annotation of 5 characters
│   ├── hero_type_manual_index.json # list for comparison with LLM (based on manual annotation results)
│   ├── Aladdin_Hero_Type_Annotation_Sheet.xlsx # manual annotation of Aladdin
│   └── ...
├── pipeline/
│   ├── build_knowledge_graph.py
│   ├── build_result_table.py
│   ├── compare_against_seal_index.py
│   ├── evaluate_framework_gemini_batch.py
│   └── evaluate_framework_two-step.py   
├── prompts/
│   ├── step1_criteria_evaluation.txt   # Step 1 prompt template
│   └── step2_evidence_extraction.txt   # Step 2 prompt template
└── README.md
```
## Installation
 
```bash
pip install google-genai vllm networkx openpyxl 
```
 
For Qwen3 evaluation, a SLURM cluster with vLLM is required. See `evaluate_framework_two-step.py` for configuration.
 
For Gemini batch evaluation, set your API key:
 
```bash
# Linux / Mac
export GEMINI_API_KEY="your_key"
 
# Windows (PowerShell)
$env:GEMINI_API_KEY="your_key"
```
 
## Usage
 
All scripts are run from the **project root**. The easiest way to run the full pipeline is through `framework_pipeline.py`:
 
```bash
python framework_pipeline.py
```
 
This runs all steps for both models in order, comparing against both Seal & White's index and manual annotations. You can control which steps, models, and ground truth to use:
 
```bash
# Run only Gemini
python framework_pipeline.py --models gemini
 
# Run only Qwen
python framework_pipeline.py --models qwen
 
# Skip evaluation (steps 1 & 2), only run post-processing
python framework_pipeline.py --skip-eval
 
# Run specific steps only
python framework_pipeline.py --steps 3 4 5
 
# Compare against Seal & White only (no manual annotations required)
python framework_pipeline.py --ground-truth seal
 
# Compare against manual annotations only
python framework_pipeline.py --ground-truth manual
 
# Combine arguments — e.g. run Gemini evaluation and comparison against Seal & White only
python framework_pipeline.py --models gemini --steps 1 3 5 --ground-truth seal
```
 
### Pipeline Steps
 
| Step | Script | Description |
|------|--------|-------------|
| 1 | `evaluate_framework_gemini_batch.py` | Gemini batch evaluation (requires `GEMINI_API_KEY`) |
| 2 | `evaluate_framework_two-step.py` | Qwen3 vLLM evaluation (requires HPC cluster with vLLM) |
| 3 | `build_result_table.py` | Aggregates assessment JSONs into a results CSV |
| 4 | `build_knowledge_graph.py` | Builds GEXF/GraphML knowledge graph for Gephi |
| 5 | `compare_against_ground_truth.py` | Compares results against Seal & White index |
| 6 | `compare_against_ground_truth.py` | Compares results against manual annotations |
 
Each pipeline script can also be run independently:
 
```bash
python pipeline/build_result_table.py --model gemini
python pipeline/build_knowledge_graph.py --model qwen
python pipeline/compare_against_ground_truth.py --model gemini --ground-truth seal-white
python pipeline/compare_against_ground_truth.py --model gemini --ground-truth manual
```
 
## Prompt Templates
 
The `prompts/` directory contains the prompt templates used in both pipeline steps. Variables in `{BRACKETS}` are filled dynamically at runtime:
 
| Variable | Description |
|---|---|
| `{CHARACTER}` | Name of the character to evaluate |
| `{STORY_TITLE}` | Title of the story |
| `{STORY_TEXT}` | Full text of the story |
| `{HERO_TYPE}` | Hero type criteria object (name, category, definition, criteria) |
| `{n_necessary}` | Number of necessary criteria for the hero type |
 
## Fit Score Categories
 
| Score | Condition |
|---|---|
| **Strong** | Necessary criteria > 60% |
| **Partial** | Necessary ≤ 60% and supporting criteria > 30% |
| **Weak** | Necessary ≤ 60% and supporting ≤ 30% (but something matched) |
| **No fit** | Necessary = 0 and supporting = 0 |
| **Disqualified** | Any exclusion criterion matched |
 

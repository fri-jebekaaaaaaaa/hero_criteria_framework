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
│   └── Hero_Type_Criteria_Framework.json  # 51 hero type definitions
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

## Prompt Templates
The prompts/ directory contains the prompt templates used in both pipeline steps. Variables in {BRACKETS} are filled dynamically at runtime:

- {CHARACTER} — name of the character to evaluate
- {STORY_TITLE} — title of the story
- {STORY_TEXT} — full text of the story
- {HERO_TYPE} — hero type criteria object (name, category, definition, criteria)
- {n_necessary} — number of necessary criteria for the hero type

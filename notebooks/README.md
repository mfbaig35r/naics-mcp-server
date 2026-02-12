# NAICS ETL Notebooks

This directory contains [marimo](https://marimo.io) notebooks for building the NAICS 2022 database from Census Bureau source files.

## Prerequisites

### Install Dependencies

```bash
# Core dependencies
pip install marimo pandas openpyxl duckdb

# For embeddings (notebook 05)
pip install sentence-transformers torch
```

### Source Data

Ensure the `raw-data/` directory contains the Census Bureau NAICS 2022 files:

```
raw-data/
├── 2-6 digit_2022_Codes.xlsx
├── 2022_NAICS_Descriptions.xlsx
├── 2022_NAICS_Index_File.xlsx
├── 2022_NAICS_Cross_References.xlsx
└── 2022_NAICS_Structure.xlsx
```

These files can be downloaded from [census.gov/naics](https://www.census.gov/naics/).

## Running the Notebooks

### Option 1: Interactive Mode (Recommended for Exploration)

Launch a notebook in the browser:

```bash
marimo edit notebooks/01_load_codes.py
```

This opens an interactive UI where you can:
- See each cell's output
- Modify and re-run cells
- Explore the data visually

### Option 2: Script Mode (Recommended for ETL)

Run a notebook as a Python script:

```bash
python notebooks/01_load_codes.py
```

This executes all cells in order and prints output to the terminal.

## Notebook Overview

Run the notebooks **in order** - each builds on the previous:

| Notebook | Purpose | Output |
|----------|---------|--------|
| `01_load_codes.py` | Load NAICS codes with hierarchy | `naics_nodes` table |
| `02_load_descriptions.py` | Add full descriptions | Updates `naics_nodes` |
| `03_load_index_terms.py` | Load 20K+ search terms | `naics_index_terms` table |
| `04_load_cross_refs.py` | Load exclusion/inclusion refs | `naics_cross_references` table |
| `05_generate_embeddings.py` | Generate vector embeddings | `naics_embeddings` table |
| `99_explore_database.py` | Interactive data explorer | (read-only) |

### Quick Start: Run All ETL

```bash
# Run the full ETL pipeline
python notebooks/01_load_codes.py
python notebooks/02_load_descriptions.py
python notebooks/03_load_index_terms.py
python notebooks/04_load_cross_refs.py
python notebooks/05_generate_embeddings_standalone.py
```

> **Note**: Use `05_generate_embeddings_standalone.py` for embeddings - it's more reliable than the marimo version due to package compatibility.

## Notebook Details

### 01_load_codes.py

Loads NAICS 2022 codes from `2-6 digit_2022_Codes.xlsx` and `2022_NAICS_Structure.xlsx`.

**What it does:**
- Parses 2,125 NAICS codes (2-6 digits)
- Derives 5-level hierarchy (sector → subsector → industry group → NAICS industry → national industry)
- Handles range codes like "31-33" (Manufacturing)
- Adds change indicators from 2017 revision

**Output:** `naics_nodes` table with columns:
- `node_code` - NAICS code (primary key)
- `level` - Hierarchy level
- `title` - Industry title
- `sector_code`, `subsector_code`, etc. - Parent codes

### 02_load_descriptions.py

Merges full descriptions from `2022_NAICS_Descriptions.xlsx`.

**What it does:**
- Adds detailed industry definitions
- Builds `raw_embedding_text` for semantic search (title + description preview)

**Output:** Updates `description` and `raw_embedding_text` columns in `naics_nodes`.

### 03_load_index_terms.py

Loads official search keywords from `2022_NAICS_Index_File.xlsx`.

**What it does:**
- Loads 20,398 index terms mapped to 6-digit codes
- Normalizes terms for search matching
- Appends top 10 terms to `raw_embedding_text`

**Output:** `naics_index_terms` table with columns:
- `term_id` - Primary key
- `naics_code` - Associated NAICS code
- `index_term` - Original term
- `term_normalized` - Lowercase, cleaned term

### 04_load_cross_refs.py

Parses cross-references from `2022_NAICS_Cross_References.xlsx`.

**What it does:**
- Loads 4,601 cross-reference entries
- Classifies as `excludes`, `see_also`, `includes`, or `general`
- Extracts target codes from reference text (92% success rate)

**Output:** `naics_cross_references` table with columns:
- `ref_id` - Primary key
- `source_code` - Code this reference belongs to
- `reference_type` - Classification of reference
- `reference_text` - Full text
- `target_code` - Extracted target code (if parseable)

### 05_generate_embeddings.py / 05_generate_embeddings_standalone.py

Generates vector embeddings for semantic search.

**What it does:**
- Uses `all-MiniLM-L6-v2` model (384 dimensions)
- Embeds title + description + index terms
- Takes ~7 seconds for all 2,125 codes

**Output:** `naics_embeddings` table with columns:
- `node_code` - NAICS code (primary key)
- `embedding` - 384-dimensional float array
- `embedding_text` - Source text that was embedded

### 99_explore_database.py

Interactive explorer for the completed database. **Launch in interactive mode:**

```bash
marimo edit notebooks/99_explore_database.py
```

**Features:**
- Browse codes by sector hierarchy
- Search index terms
- View code details (description, terms, cross-refs)
- Semantic similarity search
- Custom SQL queries

## Database Schema

After running all notebooks, the database (`data/naics.duckdb`) contains:

```
naics_nodes (2,125 rows)
├── node_code VARCHAR PRIMARY KEY
├── level VARCHAR (sector|subsector|industry_group|naics_industry|national_industry)
├── title VARCHAR
├── description TEXT
├── sector_code, subsector_code, industry_group_code, naics_industry_code VARCHAR
├── raw_embedding_text TEXT
├── change_indicator VARCHAR
└── is_trilateral BOOLEAN

naics_index_terms (20,398 rows)
├── term_id INTEGER PRIMARY KEY
├── naics_code VARCHAR
├── index_term VARCHAR
└── term_normalized VARCHAR

naics_cross_references (4,601 rows)
├── ref_id INTEGER PRIMARY KEY
├── source_code VARCHAR
├── reference_type VARCHAR
├── reference_text TEXT
└── target_code VARCHAR

naics_embeddings (2,125 rows)
├── node_code VARCHAR PRIMARY KEY
├── embedding FLOAT[384]
└── embedding_text VARCHAR
```

## Troubleshooting

### "No module named 'marimo'"

Install marimo:
```bash
pip install marimo
```

### "Data type 'str' not recognized" (DuckDB error)

This happens with pandas 3.0+ which uses `StringDtype` by default. The notebooks handle this automatically, but if you see this error, ensure you're using the latest notebook versions.

### Embeddings notebook fails with transformers error

Use the standalone script instead:
```bash
python notebooks/05_generate_embeddings_standalone.py
```

### Database is empty after running notebooks

Ensure you run notebooks **in order** (01 → 02 → 03 → 04 → 05). Each depends on the previous.

### marimo variable conflict errors

If you see "Variable X is defined in multiple cells", the notebook needs loop variables prefixed with underscore (`_var`). This is fixed in the current versions.

## Verifying the Database

Quick verification script:

```bash
python -c "
import duckdb
conn = duckdb.connect('data/naics.duckdb')
for table in ['naics_nodes', 'naics_index_terms', 'naics_cross_references', 'naics_embeddings']:
    count = conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
    print(f'{table}: {count:,} rows')
conn.close()
"
```

Expected output:
```
naics_nodes: 2,125 rows
naics_index_terms: 20,398 rows
naics_cross_references: 4,601 rows
naics_embeddings: 2,125 rows
```

## Rebuilding the Database

To rebuild from scratch:

```bash
# Remove existing database
rm data/naics.duckdb

# Run all ETL notebooks
python notebooks/01_load_codes.py
python notebooks/02_load_descriptions.py
python notebooks/03_load_index_terms.py
python notebooks/04_load_cross_refs.py
python notebooks/05_generate_embeddings_standalone.py
```

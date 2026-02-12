#!/usr/bin/env python3
"""
NAICS Index Terms Loader

Load 20,398 official index terms from Census Bureau.
Run as script: python 03_load_index_terms.py
Run interactive: marimo edit 03_load_index_terms.py
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        """
        # NAICS 2022 Index Terms Loader

        This notebook loads the official **20,398 index terms** from
        `2022_NAICS_Index_File.xlsx`.

        Index terms are:
        - Official search keywords from Census Bureau
        - Mapped to specific 6-digit NAICS codes
        - Critical for accurate classification (boost search hits)
        """
    )
    return


@app.cell
def __():
    import pandas as pd
    import duckdb
    from pathlib import Path
    import re
    return Path, duckdb, pd, re


@app.cell
def __(Path):
    # Configuration
    PROJECT_ROOT = Path(__file__).parent.parent if "__file__" in dir() else Path.cwd().parent
    RAW_DATA_DIR = PROJECT_ROOT / "raw-data"
    DB_PATH = PROJECT_ROOT / "data" / "naics.duckdb"

    print(f"Database path: {DB_PATH}")
    return DB_PATH, PROJECT_ROOT, RAW_DATA_DIR


@app.cell
def __(mo):
    mo.md("## Step 1: Load Index Terms File")
    return


@app.cell
def __(RAW_DATA_DIR, pd):
    # Load index terms
    index_file = RAW_DATA_DIR / "2022_NAICS_Index_File.xlsx"
    df_index = pd.read_excel(index_file)

    print(f"Columns: {list(df_index.columns)}")
    print(f"Loaded {len(df_index)} index terms")

    # Rename columns
    df_index = df_index.rename(columns={
        "NAICS22": "naics_code",
        "INDEX ITEM DESCRIPTION": "index_term"
    })

    # Convert code to string
    df_index["naics_code"] = df_index["naics_code"].astype(str).str.strip()

    df_index.head(10)
    return df_index, index_file


@app.cell
def __(mo):
    mo.md("## Step 2: Analyze Index Terms")
    return


@app.cell
def __(df_index):
    # Distribution of terms by code
    terms_per_code = df_index.groupby("naics_code").size()
    print("Index terms per code statistics:")
    print(terms_per_code.describe())

    print(f"\nCodes with most terms:")
    print(terms_per_code.sort_values(ascending=False).head(10))
    return (terms_per_code,)


@app.cell
def __(df_index):
    # Sample terms for a specific code
    sample_code = "445110"  # Supermarkets
    sample_terms = df_index[df_index["naics_code"] == sample_code]["index_term"].tolist()
    print(f"\nIndex terms for {sample_code} (Supermarkets):")
    for term in sample_terms[:15]:
        print(f"  - {term}")
    if len(sample_terms) > 15:
        print(f"  ... and {len(sample_terms) - 15} more")
    return sample_code, sample_terms


@app.cell
def __(mo):
    mo.md("## Step 3: Normalize Terms")
    return


@app.cell
def __(df_index, re):
    def normalize_term(term: str) -> str:
        """Normalize term for search matching."""
        if not isinstance(term, str):
            return ""
        # Lowercase
        term = term.lower()
        # Remove punctuation except hyphens
        term = re.sub(r'[^\w\s-]', '', term)
        # Collapse whitespace
        term = re.sub(r'\s+', ' ', term).strip()
        return term

    # Add normalized column
    df_index["term_normalized"] = df_index["index_term"].apply(normalize_term)

    # Add term_id
    df_index["term_id"] = range(1, len(df_index) + 1)

    print("Sample normalized terms:")
    df_index[["index_term", "term_normalized"]].head(10)
    return (normalize_term,)


@app.cell
def __(mo):
    mo.md("## Step 4: Load to Database")
    return


@app.cell
def __(DB_PATH, duckdb):
    # Connect to database
    conn = duckdb.connect(str(DB_PATH))

    # Create index terms table if not exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS naics_index_terms (
            term_id INTEGER PRIMARY KEY,
            naics_code VARCHAR NOT NULL,
            index_term VARCHAR NOT NULL,
            term_normalized VARCHAR
        )
    """)

    # Clear existing and insert
    conn.execute("DELETE FROM naics_index_terms")
    return (conn,)


@app.cell
def __(conn, df_index):
    # Prepare for insert
    df_insert = df_index[["term_id", "naics_code", "index_term", "term_normalized"]].copy()

    # Convert StringDtype to object for DuckDB compatibility (pandas 3.0)
    for col in df_insert.select_dtypes(include=["string"]).columns:
        df_insert[col] = df_insert[col].astype(object)

    # Insert
    conn.execute("INSERT INTO naics_index_terms SELECT * FROM df_insert")

    # Verify
    count = conn.execute("SELECT COUNT(*) FROM naics_index_terms").fetchone()[0]
    print(f"Loaded {count:,} index terms to database")
    return count, df_insert


@app.cell
def __(conn):
    # Create index for fast search
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_index_terms_code
        ON naics_index_terms(naics_code)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_index_terms_normalized
        ON naics_index_terms(term_normalized)
    """)

    print("Created search indexes")
    return


@app.cell
def __(mo):
    mo.md("## Step 5: Update Embedding Text with Index Terms")
    return


@app.cell
def __(conn):
    # Add top index terms to raw_embedding_text for better semantic search
    # Get top 10 index terms per code
    conn.execute("""
        WITH ranked_terms AS (
            SELECT
                naics_code,
                index_term,
                ROW_NUMBER() OVER (PARTITION BY naics_code ORDER BY LENGTH(index_term) DESC) as rn
            FROM naics_index_terms
        ),
        aggregated AS (
            SELECT
                naics_code,
                STRING_AGG(index_term, ', ' ORDER BY rn) as top_terms
            FROM ranked_terms
            WHERE rn <= 10
            GROUP BY naics_code
        )
        UPDATE naics_nodes
        SET raw_embedding_text = raw_embedding_text || ' | Index: ' || aggregated.top_terms
        FROM aggregated
        WHERE naics_nodes.node_code = aggregated.naics_code
    """)

    print("Updated embedding text with index terms")
    return


@app.cell
def __(conn):
    # Verify embedding text update
    sample = conn.execute("""
        SELECT node_code, title, LENGTH(raw_embedding_text) as embed_len
        FROM naics_nodes
        WHERE level = 'national_industry'
        ORDER BY embed_len DESC
        LIMIT 10
    """).fetchdf()
    sample
    return (sample,)


@app.cell
def __(conn):
    # Statistics
    stats = conn.execute("""
        SELECT
            COUNT(DISTINCT naics_code) as codes_with_terms,
            COUNT(*) as total_terms,
            ROUND(AVG(cnt), 1) as avg_terms_per_code
        FROM (
            SELECT naics_code, COUNT(*) as cnt
            FROM naics_index_terms
            GROUP BY naics_code
        ) t
    """).fetchdf()
    print("Index term statistics:")
    stats
    return (stats,)


@app.cell
def __(conn):
    conn.close()
    print("Database connection closed. Index terms loaded successfully!")
    return


@app.cell
def __(mo):
    mo.md(
        """
        ## Summary

        Loaded **20,398 official index terms** with:
        - Normalized versions for search matching
        - Search indexes for fast lookup
        - Updated embedding text with top terms

        **Next step:** Run `04_load_cross_refs.py` to load cross-references.
        """
    )
    return


if __name__ == "__main__":
    app.run()

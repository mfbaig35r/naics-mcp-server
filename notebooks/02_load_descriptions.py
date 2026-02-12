#!/usr/bin/env python3
"""
NAICS Description Loader

Merge full descriptions into NAICS codes table.
Run as script: python 02_load_descriptions.py
Run interactive: marimo edit 02_load_descriptions.py
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
        # NAICS 2022 Description Loader

        This notebook merges full descriptions from `2022_NAICS_Descriptions.xlsx`
        into the NAICS codes we loaded previously.

        Descriptions include:
        - Detailed industry definitions
        - "Illustrative examples" of establishments
        - Cross-reference text (what's excluded)
        """
    )
    return


@app.cell
def __():
    import pandas as pd
    import duckdb
    from pathlib import Path
    return Path, duckdb, pd


@app.cell
def __(Path):
    # Configuration
    PROJECT_ROOT = Path(__file__).parent.parent if "__file__" in dir() else Path.cwd().parent
    RAW_DATA_DIR = PROJECT_ROOT / "raw-data"
    DB_PATH = PROJECT_ROOT / "data" / "naics.duckdb"

    print(f"Database path: {DB_PATH}")
    print(f"Database exists: {DB_PATH.exists()}")
    return DB_PATH, PROJECT_ROOT, RAW_DATA_DIR


@app.cell
def __(mo):
    mo.md("## Step 1: Load Descriptions File")
    return


@app.cell
def __(RAW_DATA_DIR, pd):
    # Load descriptions
    desc_file = RAW_DATA_DIR / "2022_NAICS_Descriptions.xlsx"
    df_desc = pd.read_excel(desc_file)

    print(f"Columns: {list(df_desc.columns)}")
    print(f"Loaded {len(df_desc)} descriptions")

    # Convert code to string
    df_desc["Code"] = df_desc["Code"].astype(str).str.strip()

    df_desc.head()
    return desc_file, df_desc


@app.cell
def __(mo):
    mo.md("## Step 2: Preview Description Content")
    return


@app.cell
def __(df_desc):
    # Look at a sample description
    sample_desc = df_desc[df_desc["Code"] == "445110"].iloc[0]
    print(f"Code: {sample_desc['Code']}")
    print(f"Title: {sample_desc['Title']}")
    print(f"\nDescription preview (first 500 chars):")
    print(sample_desc['Description'][:500] if pd.notna(sample_desc['Description']) else "No description")
    return (sample_desc,)


@app.cell
def __(df_desc):
    # Check description lengths
    df_desc["desc_length"] = df_desc["Description"].fillna("").str.len()
    print("Description length statistics:")
    print(df_desc["desc_length"].describe())

    print(f"\nCodes without descriptions: {(df_desc['Description'].isna()).sum()}")
    return


@app.cell
def __(mo):
    mo.md("## Step 3: Update Database")
    return


@app.cell
def __(DB_PATH, duckdb):
    # Connect to database
    conn = duckdb.connect(str(DB_PATH))

    # Check current state
    current = conn.execute("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN description IS NOT NULL THEN 1 ELSE 0 END) as with_desc
        FROM naics_nodes
    """).fetchone()

    print(f"Current state: {current[0]} codes, {current[1]} with descriptions")
    return conn, current


@app.cell
def __(conn, df_desc):
    # Clean title in descriptions (remove trailing T)
    df_desc_clean = df_desc[["Code", "Description"]].copy()
    df_desc_clean.columns = ["node_code", "description"]

    # Convert StringDtype to object for DuckDB compatibility (pandas 3.0)
    for col in df_desc_clean.select_dtypes(include=["string"]).columns:
        df_desc_clean[col] = df_desc_clean[col].astype(object)

    # Update descriptions in database
    # DuckDB doesn't support UPDATE FROM DataFrame directly, so we use a temp table
    conn.execute("CREATE OR REPLACE TEMP TABLE temp_desc AS SELECT * FROM df_desc_clean")

    update_count = conn.execute("""
        UPDATE naics_nodes
        SET description = temp_desc.description
        FROM temp_desc
        WHERE naics_nodes.node_code = temp_desc.node_code
    """).fetchone()

    conn.execute("DROP TABLE temp_desc")

    # Verify update
    after = conn.execute("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN description IS NOT NULL THEN 1 ELSE 0 END) as with_desc
        FROM naics_nodes
    """).fetchone()

    print(f"After update: {after[0]} codes, {after[1]} with descriptions")
    return after, df_desc_clean, update_count


@app.cell
def __(mo):
    mo.md("## Step 4: Build Embedding Text")
    return


@app.cell
def __(conn):
    # Build raw_embedding_text combining title + description preview
    conn.execute("""
        UPDATE naics_nodes
        SET raw_embedding_text = title || ' | ' || COALESCE(LEFT(description, 500), '')
        WHERE raw_embedding_text IS NULL OR raw_embedding_text = ''
    """)

    # Verify
    sample = conn.execute("""
        SELECT node_code, title, LEFT(raw_embedding_text, 100) as embed_preview
        FROM naics_nodes
        WHERE level = 'national_industry'
        LIMIT 5
    """).fetchdf()
    sample
    return (sample,)


@app.cell
def __(conn):
    # Final statistics
    stats = conn.execute("""
        SELECT
            level,
            COUNT(*) as total,
            SUM(CASE WHEN description IS NOT NULL THEN 1 ELSE 0 END) as with_desc,
            ROUND(AVG(LENGTH(description)), 0) as avg_desc_len
        FROM naics_nodes
        GROUP BY level
        ORDER BY level
    """).fetchdf()

    print("Description coverage by level:")
    stats
    return (stats,)


@app.cell
def __(conn):
    conn.close()
    print("Database connection closed. Descriptions loaded successfully!")
    return


@app.cell
def __(mo):
    mo.md(
        """
        ## Summary

        Merged descriptions from `2022_NAICS_Descriptions.xlsx`:
        - Full industry definitions
        - Built `raw_embedding_text` for vector search

        **Next step:** Run `03_load_index_terms.py` to load official index terms.
        """
    )
    return


if __name__ == "__main__":
    app.run()

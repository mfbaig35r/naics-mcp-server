#!/usr/bin/env python3
"""
NAICS Code Loader

Load NAICS 2022 codes from source Excel files into DuckDB.
Run as script: python 01_load_codes.py
Run interactive: marimo edit 01_load_codes.py
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
        # NAICS 2022 Code Loader

        This notebook loads NAICS codes from the official Census Bureau Excel files
        into our DuckDB database. It handles:

        - **2-6 digit codes** from `2-6 digit_2022_Codes.xlsx`
        - **Structure metadata** from `2022_NAICS_Structure.xlsx`
        - Hierarchy derivation (sector, subsector, industry group, etc.)
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
    DATA_DIR = PROJECT_ROOT / "data"
    DB_PATH = DATA_DIR / "naics.duckdb"

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data dir: {RAW_DATA_DIR}")
    print(f"Database path: {DB_PATH}")
    return DATA_DIR, DB_PATH, PROJECT_ROOT, RAW_DATA_DIR


@app.cell
def __(mo):
    mo.md("## Step 1: Load Raw Codes")
    return


@app.cell
def __(RAW_DATA_DIR, pd):
    # Load the main codes file
    codes_file = RAW_DATA_DIR / "2-6 digit_2022_Codes.xlsx"
    df_codes_raw = pd.read_excel(codes_file)

    # Clean up - skip header row, rename columns
    df_codes_raw = df_codes_raw.dropna(subset=["2022 NAICS US   Code"])
    df_codes_raw = df_codes_raw.rename(columns={
        "Seq. No.": "seq_no",
        "2022 NAICS US   Code": "code",
        "2022 NAICS US Title": "title"
    })

    # Convert code to string (handle float formatting and range codes like "31-33")
    def clean_code(x):
        if pd.isna(x):
            return None
        if isinstance(x, (int, float)):
            return str(int(x))
        return str(x).strip()

    df_codes_raw["code"] = df_codes_raw["code"].apply(clean_code)

    print(f"Loaded {len(df_codes_raw)} codes")
    df_codes_raw.head(10)
    return codes_file, df_codes_raw


@app.cell
def __(mo):
    mo.md("## Step 2: Load Structure Metadata")
    return


@app.cell
def __(RAW_DATA_DIR, pd):
    # Load structure file for change indicators
    structure_file = RAW_DATA_DIR / "2022_NAICS_Structure.xlsx"
    df_structure_raw = pd.read_excel(structure_file, skiprows=1)

    # Rename columns
    df_structure_raw.columns = ["change_indicator", "code", "title"]
    df_structure_raw = df_structure_raw.dropna(subset=["code"])

    # Clean code column - remove "T" suffix from titles that got merged
    df_structure_raw["code"] = df_structure_raw["code"].astype(str).str.strip()

    # Clean title - remove trailing "T" (trilateral marker)
    df_structure_raw["title"] = df_structure_raw["title"].astype(str).str.rstrip("T").str.strip()

    print(f"Loaded {len(df_structure_raw)} structure entries")
    df_structure_raw.head(10)
    return df_structure_raw, structure_file


@app.cell
def __(mo):
    mo.md("## Step 3: Derive Hierarchy")
    return


@app.cell
def __(df_codes_raw, pd):
    def derive_level(code: str) -> str:
        """Derive NAICS level from code length."""
        # Handle range codes like "31-33", "44-45", "48-49"
        if "-" in code:
            return "sector"
        length = len(code)
        levels = {
            2: "sector",
            3: "subsector",
            4: "industry_group",
            5: "naics_industry",
            6: "national_industry"
        }
        return levels.get(length, "unknown")

    def derive_hierarchy(code: str) -> dict:
        """Derive parent codes at each level."""
        # Handle range codes like "31-33" - they are their own sector
        if "-" in code:
            return {
                "sector_code": code,
                "subsector_code": None,
                "industry_group_code": None,
                "naics_industry_code": None,
            }
        return {
            "sector_code": code[:2] if len(code) >= 2 else None,
            "subsector_code": code[:3] if len(code) >= 3 else None,
            "industry_group_code": code[:4] if len(code) >= 4 else None,
            "naics_industry_code": code[:5] if len(code) >= 5 else None,
        }

    # Build the codes dataframe with hierarchy
    df_codes = df_codes_raw[["code", "title"]].copy()
    df_codes["level"] = df_codes["code"].apply(derive_level)

    # Add hierarchy columns
    hierarchy_df = df_codes["code"].apply(derive_hierarchy).apply(pd.Series)
    df_codes = pd.concat([df_codes, hierarchy_df], axis=1)

    # Clean title - remove trailing "T" (trilateral marker)
    df_codes["title"] = df_codes["title"].str.rstrip("T").str.strip()

    print(f"\nCodes by level:")
    print(df_codes["level"].value_counts().sort_index())
    df_codes.head(10)
    return derive_hierarchy, derive_level, df_codes, hierarchy_df


@app.cell
def __(df_codes, df_structure_raw):
    # Merge in change indicators from structure file
    structure_lookup = df_structure_raw[["code", "change_indicator"]].drop_duplicates()
    df_final = df_codes.merge(structure_lookup, on="code", how="left")

    # Add trilateral flag (all NAICS codes are trilateral by default)
    df_final["is_trilateral"] = True

    print(f"\nChange indicators found:")
    print(df_final["change_indicator"].value_counts(dropna=False))
    df_final.head(10)
    return df_final, structure_lookup


@app.cell
def __(mo):
    mo.md("## Step 4: Load to Database")
    return


@app.cell
def __(DB_PATH, df_final, duckdb):
    # Connect to database
    conn = duckdb.connect(str(DB_PATH))

    # Create schema if not exists (from our database.py)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS naics_nodes (
            node_code VARCHAR PRIMARY KEY,
            level VARCHAR NOT NULL,
            title VARCHAR NOT NULL,
            description TEXT,
            sector_code VARCHAR,
            subsector_code VARCHAR,
            industry_group_code VARCHAR,
            naics_industry_code VARCHAR,
            raw_embedding_text TEXT,
            change_indicator VARCHAR,
            is_trilateral BOOLEAN DEFAULT true
        )
    """)

    # Clear existing data and insert
    conn.execute("DELETE FROM naics_nodes")

    # Prepare dataframe for insert
    df_insert = df_final.rename(columns={"code": "node_code"})
    df_insert = df_insert[[
        "node_code", "level", "title", "sector_code", "subsector_code",
        "industry_group_code", "naics_industry_code", "change_indicator", "is_trilateral"
    ]]

    # Convert dtypes for DuckDB compatibility (pandas 3.0 uses StringDtype by default)
    for col in df_insert.select_dtypes(include=["string"]).columns:
        df_insert[col] = df_insert[col].astype(object)
    df_insert["is_trilateral"] = df_insert["is_trilateral"].astype(bool)

    # Insert using DuckDB's DataFrame integration with explicit columns
    conn.execute("""
        INSERT INTO naics_nodes (
            node_code, level, title, sector_code, subsector_code,
            industry_group_code, naics_industry_code, change_indicator, is_trilateral
        )
        SELECT * FROM df_insert
    """)

    # Verify
    result = conn.execute("SELECT level, COUNT(*) as cnt FROM naics_nodes GROUP BY level ORDER BY level").fetchall()
    print("Loaded to database:")
    for level, cnt in result:
        print(f"  {level}: {cnt}")

    total = conn.execute("SELECT COUNT(*) FROM naics_nodes").fetchone()[0]
    print(f"\nTotal codes: {total}")
    return conn, df_insert, result, total


@app.cell
def __(conn):
    # Sample verification
    sample = conn.execute("""
        SELECT node_code, level, title, sector_code, change_indicator
        FROM naics_nodes
        WHERE level = 'national_industry'
        LIMIT 10
    """).fetchdf()
    sample
    return (sample,)


@app.cell
def __(conn):
    # Close connection
    conn.close()
    print("Database connection closed. Codes loaded successfully!")
    return


@app.cell
def __(mo):
    mo.md(
        """
        ## Summary

        Loaded NAICS 2022 codes with:
        - 5-level hierarchy (Sector → National Industry)
        - Change indicators from 2017 revision
        - Trilateral agreement flags

        **Next step:** Run `02_load_descriptions.py` to add full descriptions.
        """
    )
    return


if __name__ == "__main__":
    app.run()

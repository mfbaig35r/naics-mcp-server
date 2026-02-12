#!/usr/bin/env python3
"""
NAICS Cross-References Loader

Load and parse cross-references (exclusions/inclusions) for classification guidance.
Run as script: python 04_load_cross_refs.py
Run interactive: marimo edit 04_load_cross_refs.py
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
        # NAICS 2022 Cross-References Loader

        This notebook loads cross-references from `2022_NAICS_Cross_References.xlsx`.

        Cross-references are **critical for classification accuracy**:
        - They tell you what's **excluded** from a code
        - They point to the **correct code** for excluded activities
        - Example: "Retail bakeries" are excluded from Manufacturing (311) → classified in Retail (445)
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
    mo.md("## Step 1: Load Cross-References File")
    return


@app.cell
def __(RAW_DATA_DIR, pd):
    # Load cross-references
    xref_file = RAW_DATA_DIR / "2022_NAICS_Cross_References.xlsx"
    df_xref = pd.read_excel(xref_file)

    print(f"Columns: {list(df_xref.columns)}")
    print(f"Loaded {len(df_xref)} cross-references")

    # Rename columns
    df_xref = df_xref.rename(columns={
        "Code": "source_code",
        "Cross-Reference": "reference_text"
    })

    # Convert code to string
    df_xref["source_code"] = df_xref["source_code"].astype(str).str.strip()

    df_xref.head()
    return df_xref, xref_file


@app.cell
def __(mo):
    mo.md("## Step 2: Analyze Cross-Reference Patterns")
    return


@app.cell
def __(df_xref):
    # Look at sample cross-references
    print("Sample cross-references:\n")
    for _, row in df_xref.head(5).iterrows():
        print(f"Code {row['source_code']}:")
        print(f"  {row['reference_text'][:200]}...")
        print()
    return


@app.cell
def __(df_xref, re):
    # Pattern analysis - what types of references exist?
    def detect_reference_type(text: str) -> str:
        """Detect the type of cross-reference."""
        if not isinstance(text, str):
            return "unknown"
        text_lower = text.lower()

        if "are classified in" in text_lower:
            return "excludes"
        elif "see industry" in text_lower or "see " in text_lower:
            return "see_also"
        elif "includes" in text_lower:
            return "includes"
        else:
            return "general"

    df_xref["reference_type"] = df_xref["reference_text"].apply(detect_reference_type)

    print("Reference types:")
    print(df_xref["reference_type"].value_counts())
    return (detect_reference_type,)


@app.cell
def __(mo):
    mo.md("## Step 3: Parse Target Codes")
    return


@app.cell
def __(df_xref, re):
    def extract_target_codes(text: str) -> list:
        """Extract NAICS codes mentioned in cross-reference text."""
        if not isinstance(text, str):
            return []

        # Pattern: "Industry XXXXXX" or "U.S. Industry XXXXXX"
        patterns = [
            r'Industry\s+(\d{6})',
            r'Industry\s+(\d{5})',
            r'Industry\s+(\d{4})',
            r'Subsector\s+(\d{3})',
            r'Sector\s+(\d{2})',
            r'classified in\s+(\d{6})',
            r'classified in\s+(\d{5})',
        ]

        codes = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            codes.extend(matches)

        return list(set(codes))  # Deduplicate

    # Extract target codes
    df_xref["target_codes"] = df_xref["reference_text"].apply(extract_target_codes)

    # Flatten to primary target (first mentioned)
    df_xref["target_code"] = df_xref["target_codes"].apply(
        lambda x: x[0] if x else None
    )

    print("Sample parsed references:")
    df_xref[["source_code", "reference_type", "target_code", "reference_text"]].head(10)
    return (extract_target_codes,)


@app.cell
def __(df_xref):
    # Statistics on target code extraction
    has_target = df_xref["target_code"].notna().sum()
    total = len(df_xref)
    print(f"Cross-references with parsed target codes: {has_target}/{total} ({has_target/total:.1%})")

    # Show some that we couldn't parse
    unparsed = df_xref[df_xref["target_code"].isna()]["reference_text"].head(5).tolist()
    print("\nSamples without parsed target codes:")
    for text in unparsed:
        print(f"  - {text[:100]}...")
    return has_target, total, unparsed


@app.cell
def __(mo):
    mo.md("## Step 4: Load to Database")
    return


@app.cell
def __(DB_PATH, duckdb):
    # Connect to database
    conn = duckdb.connect(str(DB_PATH))

    # Create cross-references table (drop first to ensure clean schema)
    conn.execute("DROP TABLE IF EXISTS naics_cross_references")
    conn.execute("""
        CREATE TABLE naics_cross_references (
            ref_id INTEGER PRIMARY KEY,
            source_code VARCHAR NOT NULL,
            reference_type VARCHAR,
            reference_text TEXT NOT NULL,
            target_code VARCHAR
        )
    """)
    return (conn,)


@app.cell
def __(conn, df_xref):
    # Prepare for insert
    df_insert = df_xref[["source_code", "reference_type", "reference_text", "target_code"]].copy()
    df_insert["ref_id"] = range(1, len(df_insert) + 1)
    df_insert = df_insert[["ref_id", "source_code", "reference_type", "reference_text", "target_code"]]

    # Convert StringDtype to object for DuckDB compatibility (pandas 3.0)
    for col in df_insert.select_dtypes(include=["string"]).columns:
        df_insert[col] = df_insert[col].astype(object)

    # Insert
    conn.execute("INSERT INTO naics_cross_references SELECT * FROM df_insert")

    # Verify
    count = conn.execute("SELECT COUNT(*) FROM naics_cross_references").fetchone()[0]
    print(f"Loaded {count:,} cross-references to database")
    return count, df_insert


@app.cell
def __(conn):
    # Create indexes
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_xref_source
        ON naics_cross_references(source_code)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_xref_target
        ON naics_cross_references(target_code)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_xref_type
        ON naics_cross_references(reference_type)
    """)

    print("Created search indexes")
    return


@app.cell
def __(mo):
    mo.md("## Step 5: Verify Cross-Reference Lookups")
    return


@app.cell
def __(conn):
    # Test: Get exclusions for a specific code
    test_code = "311811"  # Retail Bakeries
    exclusions = conn.execute(f"""
        SELECT source_code, reference_type, target_code,
               LEFT(reference_text, 100) as text_preview
        FROM naics_cross_references
        WHERE source_code = '{test_code}'
    """).fetchdf()

    print(f"Cross-references for {test_code}:")
    exclusions
    return exclusions, test_code


@app.cell
def __(conn):
    # Test: Find all codes that reference a target
    target = "445110"  # Supermarkets
    incoming = conn.execute(f"""
        SELECT source_code, reference_type,
               LEFT(reference_text, 80) as text_preview
        FROM naics_cross_references
        WHERE target_code = '{target}'
        LIMIT 10
    """).fetchdf()

    print(f"Codes that reference {target} (Supermarkets):")
    incoming
    return incoming, target


@app.cell
def __(conn):
    # Statistics by reference type
    stats = conn.execute("""
        SELECT
            reference_type,
            COUNT(*) as count,
            SUM(CASE WHEN target_code IS NOT NULL THEN 1 ELSE 0 END) as with_target
        FROM naics_cross_references
        GROUP BY reference_type
        ORDER BY count DESC
    """).fetchdf()
    print("Cross-reference statistics:")
    stats
    return (stats,)


@app.cell
def __(conn):
    conn.close()
    print("Database connection closed. Cross-references loaded successfully!")
    return


@app.cell
def __(mo):
    mo.md(
        """
        ## Summary

        Loaded cross-references with:
        - Reference type classification (excludes, see_also, includes)
        - Parsed target codes for linked navigation
        - Indexed for fast lookup

        **Cross-references are critical for classification accuracy!**
        When a query matches an exclusion, we should:
        1. Warn the user
        2. Suggest the correct target code

        **Next step:** Run `05_generate_embeddings.py` to create vector embeddings.
        """
    )
    return


if __name__ == "__main__":
    app.run()

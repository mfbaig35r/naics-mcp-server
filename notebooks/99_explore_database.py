#!/usr/bin/env python3
"""
NAICS Database Explorer

Interactive exploration of the NAICS database.
Run interactive: marimo edit 99_explore_database.py
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
        # NAICS Database Explorer

        Interactive exploration of the NAICS 2022 database.
        Use this notebook to:

        - Browse codes by hierarchy
        - Search index terms
        - View cross-references
        - Test semantic search
        """
    )
    return


@app.cell
def __():
    import duckdb
    import pandas as pd
    import numpy as np
    from pathlib import Path
    return Path, duckdb, np, pd


@app.cell
def __(Path):
    # Configuration
    PROJECT_ROOT = Path(__file__).parent.parent if "__file__" in dir() else Path.cwd().parent
    DB_PATH = PROJECT_ROOT / "data" / "naics.duckdb"

    print(f"Database: {DB_PATH}")
    print(f"Exists: {DB_PATH.exists()}")
    return DB_PATH, PROJECT_ROOT


@app.cell
def __(DB_PATH, duckdb):
    # Connect to database
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    print("Connected to database (read-only)")
    return (conn,)


@app.cell
def __(mo):
    mo.md("## Database Statistics")
    return


@app.cell
def __(conn):
    # Overall statistics
    stats = conn.execute("""
        SELECT
            (SELECT COUNT(*) FROM naics_nodes) as total_codes,
            (SELECT COUNT(*) FROM naics_index_terms) as index_terms,
            (SELECT COUNT(*) FROM naics_cross_references) as cross_refs,
            (SELECT COUNT(*) FROM naics_embeddings) as embeddings
    """).fetchdf()

    print("Database contents:")
    for col in stats.columns:
        print(f"  {col}: {stats[col].iloc[0]:,}")
    return (stats,)


@app.cell
def __(conn):
    # Codes by level
    levels = conn.execute("""
        SELECT level, COUNT(*) as count
        FROM naics_nodes
        GROUP BY level
        ORDER BY
            CASE level
                WHEN 'sector' THEN 1
                WHEN 'subsector' THEN 2
                WHEN 'industry_group' THEN 3
                WHEN 'naics_industry' THEN 4
                WHEN 'national_industry' THEN 5
            END
    """).fetchdf()
    levels
    return (levels,)


@app.cell
def __(mo):
    mo.md("## Browse Hierarchy")
    return


@app.cell
def __(mo):
    # Sector selector
    sector_input = mo.ui.dropdown(
        options={
            "11": "11 - Agriculture",
            "21": "21 - Mining",
            "22": "22 - Utilities",
            "23": "23 - Construction",
            "31": "31-33 - Manufacturing",
            "42": "42 - Wholesale Trade",
            "44": "44-45 - Retail Trade",
            "48": "48-49 - Transportation",
            "51": "51 - Information",
            "52": "52 - Finance/Insurance",
            "53": "53 - Real Estate",
            "54": "54 - Professional Services",
            "55": "55 - Management",
            "56": "56 - Admin/Support",
            "61": "61 - Education",
            "62": "62 - Health Care",
            "71": "71 - Arts/Entertainment",
            "72": "72 - Accommodation/Food",
            "81": "81 - Other Services",
            "92": "92 - Public Administration",
        },
        value="44",
        label="Select Sector"
    )
    sector_input
    return (sector_input,)


@app.cell
def __(conn, sector_input):
    # Show hierarchy for selected sector
    selected_sector = sector_input.value

    hierarchy = conn.execute(f"""
        SELECT
            node_code,
            level,
            title,
            REPEAT('  ', LENGTH(node_code) - 2) || title as indented_title
        FROM naics_nodes
        WHERE sector_code = '{selected_sector}'
        ORDER BY node_code
        LIMIT 50
    """).fetchdf()

    print(f"Hierarchy for Sector {selected_sector}:")
    hierarchy[["node_code", "level", "indented_title"]]
    return hierarchy, selected_sector


@app.cell
def __(mo):
    mo.md("## Search Index Terms")
    return


@app.cell
def __(mo):
    search_input = mo.ui.text(
        value="grocery",
        label="Search index terms",
        placeholder="Enter search term..."
    )
    search_input
    return (search_input,)


@app.cell
def __(conn, search_input):
    # Search index terms
    search_term = search_input.value.lower()

    if search_term:
        results = conn.execute(f"""
            SELECT
                t.naics_code,
                n.title,
                t.index_term
            FROM naics_index_terms t
            JOIN naics_nodes n ON t.naics_code = n.node_code
            WHERE t.term_normalized LIKE '%{search_term}%'
            LIMIT 20
        """).fetchdf()

        print(f"Index terms matching '{search_term}':")
        results
    else:
        print("Enter a search term above")
        results = None
    return results, search_term


@app.cell
def __(mo):
    mo.md("## View Code Details")
    return


@app.cell
def __(mo):
    code_input = mo.ui.text(
        value="445110",
        label="Enter NAICS code",
        placeholder="e.g., 445110"
    )
    code_input
    return (code_input,)


@app.cell
def __(code_input, conn):
    # Get code details
    code = code_input.value.strip()

    if code:
        detail = conn.execute(f"""
            SELECT *
            FROM naics_nodes
            WHERE node_code = '{code}'
        """).fetchdf()

        if len(detail) > 0:
            row = detail.iloc[0]
            print(f"Code: {row['node_code']}")
            print(f"Title: {row['title']}")
            print(f"Level: {row['level']}")
            print(f"\nHierarchy:")
            print(f"  Sector: {row['sector_code']}")
            print(f"  Subsector: {row['subsector_code']}")
            print(f"  Industry Group: {row['industry_group_code']}")
            print(f"  NAICS Industry: {row['naics_industry_code']}")
            print(f"\nDescription preview:")
            desc = row['description']
            if desc:
                print(f"  {desc[:500]}...")
        else:
            print(f"Code {code} not found")
    return code, detail, row


@app.cell
def __(code, conn):
    # Get index terms for code
    if code:
        terms = conn.execute(f"""
            SELECT index_term
            FROM naics_index_terms
            WHERE naics_code = '{code}'
        """).fetchdf()

        if len(terms) > 0:
            print(f"\nIndex terms for {code}:")
            for t in terms["index_term"].tolist()[:15]:
                print(f"  - {t}")
            if len(terms) > 15:
                print(f"  ... and {len(terms) - 15} more")
    return t, terms


@app.cell
def __(code, conn):
    # Get cross-references for code
    if code:
        xrefs = conn.execute(f"""
            SELECT reference_type, target_code, reference_text
            FROM naics_cross_references
            WHERE source_code = '{code}'
        """).fetchdf()

        if len(xrefs) > 0:
            print(f"\nCross-references for {code}:")
            for _, xr in xrefs.iterrows():
                print(f"  [{xr['reference_type']}] → {xr['target_code'] or 'N/A'}")
                print(f"    {xr['reference_text'][:100]}...")
        else:
            print(f"\nNo cross-references for {code}")
    return xr, xrefs


@app.cell
def __(mo):
    mo.md("## Semantic Search (requires embeddings)")
    return


@app.cell
def __(mo):
    semantic_input = mo.ui.text(
        value="selling fresh produce and groceries",
        label="Semantic search query",
        placeholder="Describe a business activity..."
    )
    semantic_input
    return (semantic_input,)


@app.cell
def __(conn, np, semantic_input):
    # Check if embeddings exist and model available
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")

        emb_count = conn.execute("SELECT COUNT(*) FROM naics_embeddings").fetchone()[0]

        if emb_count > 0 and semantic_input.value:
            query = semantic_input.value
            query_emb = model.encode(query)

            # Get embeddings and compute similarity
            all_emb = conn.execute("""
                SELECT e.node_code, e.embedding, n.title, n.level
                FROM naics_embeddings e
                JOIN naics_nodes n ON e.node_code = n.node_code
            """).fetchall()

            sims = []
            for code, emb, title, level in all_emb:
                emb_arr = np.array(emb)
                sim = np.dot(query_emb, emb_arr) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(emb_arr)
                )
                sims.append((code, title, level, sim))

            sims.sort(key=lambda x: x[3], reverse=True)

            print(f"Semantic search: '{query}'\n")
            print("Top 10 matches:")
            for code, title, level, sim in sims[:10]:
                print(f"  [{code}] {title[:45]:45s} ({level:17s}) sim={sim:.3f}")
        elif emb_count == 0:
            print("No embeddings found. Run 05_generate_embeddings.py first.")
        else:
            print("Enter a search query above")

    except ImportError:
        print("sentence-transformers not installed. Install with: pip install sentence-transformers")
    return (
        all_emb,
        code,
        emb,
        emb_arr,
        emb_count,
        level,
        model,
        query,
        query_emb,
        sim,
        sims,
        title,
    )


@app.cell
def __(mo):
    mo.md("## Custom SQL Query")
    return


@app.cell
def __(mo):
    sql_input = mo.ui.text_area(
        value="SELECT level, COUNT(*) as cnt\nFROM naics_nodes\nGROUP BY level\nORDER BY cnt DESC",
        label="SQL Query (read-only)",
        rows=5
    )
    sql_input
    return (sql_input,)


@app.cell
def __(conn, sql_input):
    # Execute custom query
    sql = sql_input.value.strip()

    if sql:
        try:
            result = conn.execute(sql).fetchdf()
            print(f"Results ({len(result)} rows):")
            result
        except Exception as e:
            print(f"Error: {e}")
    return result, sql


@app.cell
def __(mo):
    mo.md(
        """
        ## Tips

        **Useful queries:**

        ```sql
        -- Find codes by title pattern
        SELECT node_code, title FROM naics_nodes
        WHERE title ILIKE '%software%'

        -- Get full hierarchy path
        SELECT n.*, s.title as sector_title
        FROM naics_nodes n
        JOIN naics_nodes s ON n.sector_code = s.node_code
        WHERE n.node_code = '541511'

        -- Cross-reference network
        SELECT source_code, target_code, reference_type
        FROM naics_cross_references
        WHERE target_code IS NOT NULL
        ```
        """
    )
    return


if __name__ == "__main__":
    app.run()

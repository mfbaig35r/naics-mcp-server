#!/usr/bin/env python3
"""
NAICS Semantic Relationship Builder

Pre-compute semantic similarity relationships between all NAICS codes
using FAISS for efficient similarity computation.

Run as script: python 06_compute_relationships.py
Run interactive: marimo edit 06_compute_relationships.py

This creates the `naics_relationships` table with pre-computed:
- Same-sector alternatives (codes in the same sector)
- Cross-sector alternatives (similar codes in different sectors)
- Relationship statistics

Adapted from local-frontier/clustering/index.py pattern.
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
        # NAICS Semantic Relationship Builder

        This notebook pre-computes **semantic similarity relationships** between all
        NAICS codes, enabling:

        - Cross-granularity relationships (6-digit code similar to different 2-digit sector)
        - Same-sector alternatives (within sector)
        - Cross-sector alternatives (across sectors)

        Uses **FAISS IndexFlatIP** on L2-normalized embeddings for cosine similarity.

        ## Parameters
        - `MIN_SIMILARITY`: 0.70 (include relationships above 70%)
        - `MAX_NEIGHBORS`: 50 (check top 50 neighbors per code)
        """
    )
    return


@app.cell
def __():
    import duckdb
    import numpy as np
    import json
    from pathlib import Path
    from dataclasses import dataclass
    from typing import Iterator
    import time

    try:
        import faiss
    except ImportError:
        print("FAISS not installed. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "faiss-cpu"])
        import faiss

    return Path, dataclass, duckdb, faiss, json, np, time, Iterator


@app.cell
def __(Path):
    # Configuration
    PROJECT_ROOT = Path(__file__).parent.parent if "__file__" in dir() else Path.cwd().parent
    DB_PATH = PROJECT_ROOT / "data" / "naics.duckdb"

    # Similarity parameters
    MIN_SIMILARITY = 0.70  # Include relationships above 70%
    MAX_NEIGHBORS = 50     # Check top 50 neighbors per code
    EMBEDDING_DIM = 384    # all-MiniLM-L6-v2 dimension

    print(f"Database path: {DB_PATH}")
    print(f"Min similarity: {MIN_SIMILARITY}")
    print(f"Max neighbors: {MAX_NEIGHBORS}")
    return DB_PATH, EMBEDDING_DIM, MAX_NEIGHBORS, MIN_SIMILARITY, PROJECT_ROOT


@app.cell
def __(mo):
    mo.md("## Step 1: Load Embeddings and Metadata")
    return


@app.cell
def __(DB_PATH, duckdb, np):
    # Connect to database
    conn = duckdb.connect(str(DB_PATH))

    # Load embeddings with metadata
    df = conn.execute("""
        SELECT
            e.node_code,
            e.embedding,
            n.level,
            n.title,
            n.sector_code,
            n.subsector_code,
            n.industry_group_code,
            n.naics_industry_code
        FROM naics_embeddings e
        JOIN naics_nodes n ON e.node_code = n.node_code
        ORDER BY e.node_code
    """).fetchdf()

    print(f"Loaded {len(df)} codes with embeddings")

    # Extract embeddings as numpy array
    embeddings = np.array([np.array(e) for e in df["embedding"].values])
    codes = df["node_code"].tolist()

    # Build metadata dict
    metadata = {}
    for _, row in df.iterrows():
        metadata[row["node_code"]] = {
            "level": row["level"],
            "title": row["title"],
            "sector_code": row["sector_code"],
            "subsector_code": row["subsector_code"],
            "industry_group_code": row["industry_group_code"],
            "naics_industry_code": row["naics_industry_code"],
            "hierarchy_path": [
                c for c in [
                    row["sector_code"],
                    row["subsector_code"],
                    row["industry_group_code"],
                    row["naics_industry_code"],
                    row["node_code"] if row["level"] == "national_industry" else None
                ] if c is not None
            ]
        }

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Metadata entries: {len(metadata)}")
    return codes, conn, df, embeddings, metadata


@app.cell
def __(mo):
    mo.md("## Step 2: Build FAISS Index")
    return


@app.cell
def __(EMBEDDING_DIM, codes, embeddings, faiss, np, time):
    # Normalize embeddings for cosine similarity
    _start = time.time()

    embeddings_normalized = embeddings.astype(np.float32).copy()
    faiss.normalize_L2(embeddings_normalized)

    # Build FAISS index with inner product (cosine after normalization)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings_normalized)

    # Build lookup mappings
    code_to_position = {code: i for i, code in enumerate(codes)}
    position_to_code = {i: code for i, code in enumerate(codes)}

    _elapsed = time.time() - _start
    print(f"FAISS index built in {_elapsed:.2f}s")
    print(f"Index size: {index.ntotal} vectors")
    return code_to_position, embeddings_normalized, index, position_to_code


@app.cell
def __(mo):
    mo.md("## Step 3: Compute All Relationships")
    return


@app.cell
def __(
    MAX_NEIGHBORS,
    MIN_SIMILARITY,
    codes,
    embeddings_normalized,
    index,
    metadata,
    np,
    time,
):
    def compute_all_relationships(
        embeddings: np.ndarray,
        codes_list: list[str],
        metadata_dict: dict,
        faiss_index,
        min_similarity: float = 0.70,
        max_neighbors: int = 50
    ) -> dict[str, dict]:
        """
        Compute relationships for all codes.

        Returns a dict mapping code -> relationship document.
        """
        results = {}

        # Batch query all codes at once
        k = min(max_neighbors, len(codes_list))
        similarities, indices = faiss_index.search(embeddings, k)

        for i, code in enumerate(codes_list):
            meta = metadata_dict[code]
            same_sector = []
            cross_sector = []

            for sim, j in zip(similarities[i], indices[i]):
                if i == j:  # Skip self
                    continue
                if sim < min_similarity:
                    continue

                target_code = codes_list[j]
                target_meta = metadata_dict[target_code]

                alt = {
                    "target_code": target_code,
                    "target_title": target_meta["title"],
                    "target_level": target_meta["level"],
                    "similarity_score": round(float(sim), 4),
                    "target_sector": target_meta["sector_code"],
                    "target_hierarchy": target_meta["hierarchy_path"]
                }

                # Classify by sector
                if meta["sector_code"] == target_meta["sector_code"]:
                    same_sector.append(alt)
                else:
                    # Add relationship note for cross-sector
                    alt["relationship_note"] = (
                        f"Cross-sector: {meta['sector_code']} -> {target_meta['sector_code']}"
                    )
                    cross_sector.append(alt)

            # Sort by similarity score descending
            same_sector.sort(key=lambda x: x["similarity_score"], reverse=True)
            cross_sector.sort(key=lambda x: x["similarity_score"], reverse=True)

            # Compute statistics
            all_alts = same_sector + cross_sector
            stats = {
                "same_sector_count": len(same_sector),
                "cross_sector_count": len(cross_sector),
                "total_count": len(all_alts),
                "max_similarity": max([a["similarity_score"] for a in all_alts], default=0),
                "avg_similarity": round(
                    float(np.mean([a["similarity_score"] for a in all_alts])), 4
                ) if all_alts else 0
            }

            results[code] = {
                "node_code": code,
                "level": meta["level"],
                "title": meta["title"],
                "sector_code": meta["sector_code"],
                "hierarchy_path": meta["hierarchy_path"],
                "same_sector_alternatives": same_sector,
                "cross_sector_alternatives": cross_sector,
                "relationship_stats": stats
            }

        return results

    # Compute relationships
    print(f"Computing relationships for {len(codes)} codes...")
    _start = time.time()

    relationships = compute_all_relationships(
        embeddings_normalized,
        codes,
        metadata,
        index,
        min_similarity=MIN_SIMILARITY,
        max_neighbors=MAX_NEIGHBORS
    )

    _elapsed = time.time() - _start
    print(f"Computed relationships in {_elapsed:.1f}s")
    return compute_all_relationships, relationships


@app.cell
def __(mo):
    mo.md("## Step 4: Analyze Results")
    return


@app.cell
def __(relationships, np):
    # Analyze relationship statistics
    total_same_sector = sum(
        r["relationship_stats"]["same_sector_count"]
        for r in relationships.values()
    )
    total_cross_sector = sum(
        r["relationship_stats"]["cross_sector_count"]
        for r in relationships.values()
    )

    codes_with_cross_sector = sum(
        1 for r in relationships.values()
        if r["relationship_stats"]["cross_sector_count"] > 0
    )

    avg_relationships = np.mean([
        r["relationship_stats"]["total_count"]
        for r in relationships.values()
    ])

    print("=== Relationship Statistics ===")
    print(f"Total codes: {len(relationships)}")
    print(f"Total same-sector relationships: {total_same_sector}")
    print(f"Total cross-sector relationships: {total_cross_sector}")
    print(f"Codes with cross-sector relationships: {codes_with_cross_sector}")
    print(f"Average relationships per code: {avg_relationships:.1f}")
    return (
        avg_relationships,
        codes_with_cross_sector,
        total_cross_sector,
        total_same_sector,
    )


@app.cell
def __(relationships):
    # Sample: Look at cross-sector relationships for a restaurant code
    sample_code = "722511"  # Full-Service Restaurants
    if sample_code in relationships:
        sample = relationships[sample_code]
        print(f"\n=== Sample: {sample_code} - {sample['title']} ===")
        print(f"Level: {sample['level']}")
        print(f"Sector: {sample['sector_code']}")
        print(f"\nSame-sector alternatives (top 5):")
        for _alt in sample["same_sector_alternatives"][:5]:
            print(f"  [{_alt['target_code']}] {_alt['target_title'][:40]:40s} ({_alt['similarity_score']:.2f})")

        print(f"\nCross-sector alternatives (top 5):")
        for _alt in sample["cross_sector_alternatives"][:5]:
            print(f"  [{_alt['target_code']}] {_alt['target_title'][:40]:40s} ({_alt['similarity_score']:.2f})")
            print(f"    {_alt.get('relationship_note', '')}")
    return (sample, sample_code)


@app.cell
def __(mo):
    mo.md("## Step 5: Store Relationships in Database")
    return


@app.cell
def __(conn):
    # Create relationships table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS naics_relationships (
            node_code VARCHAR PRIMARY KEY,
            json_data JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Clear existing data
    conn.execute("DELETE FROM naics_relationships")
    print("Relationships table ready")
    return


@app.cell
def __(conn, json, relationships, time):
    # Insert relationships
    print(f"Inserting {len(relationships)} relationship records...")
    _start = time.time()

    for _idx, (_code, _rel) in enumerate(relationships.items()):
        _json_str = json.dumps(_rel)
        conn.execute(
            "INSERT INTO naics_relationships (node_code, json_data) VALUES (?, ?)",
            [_code, _json_str]
        )
        if (_idx + 1) % 500 == 0:
            print(f"  Inserted {_idx + 1}/{len(relationships)}")

    _elapsed = time.time() - _start
    print(f"Inserted all relationships in {_elapsed:.1f}s")
    return


@app.cell
def __(conn):
    # Verify insertion
    _count = conn.execute("SELECT COUNT(*) FROM naics_relationships").fetchone()[0]
    print(f"Total relationships in database: {_count}")

    # Sample query
    _sample = conn.execute("""
        SELECT
            node_code,
            json_extract_string(json_data, '$.title') as title,
            json_extract(json_data, '$.relationship_stats.same_sector_count') as same_sector,
            json_extract(json_data, '$.relationship_stats.cross_sector_count') as cross_sector
        FROM naics_relationships
        ORDER BY json_extract(json_data, '$.relationship_stats.cross_sector_count') DESC
        LIMIT 10
    """).fetchdf()

    print("\nTop 10 codes by cross-sector relationships:")
    _sample
    return


@app.cell
def __(mo):
    mo.md("## Step 6: Test Relationship Queries")
    return


@app.cell
def __(conn, json):
    def get_similar_codes(code: str, min_similarity: float = 0.75, include_cross_sector: bool = True) -> dict:
        """Test query: Get pre-computed similar codes."""
        result = conn.execute("""
            SELECT json_data
            FROM naics_relationships
            WHERE node_code = ?
        """, [code]).fetchone()

        if not result:
            return {"error": f"Code {code} not found"}

        data = json.loads(result[0])

        # Filter by minimum similarity
        same_sector = [
            alt for alt in data["same_sector_alternatives"]
            if alt["similarity_score"] >= min_similarity
        ]

        if include_cross_sector:
            cross_sector = [
                alt for alt in data["cross_sector_alternatives"]
                if alt["similarity_score"] >= min_similarity
            ]
        else:
            cross_sector = []

        return {
            "node_code": code,
            "title": data["title"],
            "level": data["level"],
            "same_sector_alternatives": same_sector,
            "cross_sector_alternatives": cross_sector,
            "filtered_stats": {
                "same_sector_count": len(same_sector),
                "cross_sector_count": len(cross_sector)
            }
        }

    # Test the query function
    test_result = get_similar_codes("541511", min_similarity=0.75)  # Custom Software
    print(f"\n=== Query Test: 541511 ===")
    print(f"Title: {test_result.get('title', 'N/A')}")
    print(f"Same-sector alternatives: {test_result.get('filtered_stats', {}).get('same_sector_count', 0)}")
    print(f"Cross-sector alternatives: {test_result.get('filtered_stats', {}).get('cross_sector_count', 0)}")

    if test_result.get("cross_sector_alternatives"):
        print("\nCross-sector alternatives:")
        for _alt in test_result["cross_sector_alternatives"][:3]:
            print(f"  [{_alt['target_code']}] {_alt['target_title'][:50]} ({_alt['similarity_score']:.2f})")
    return get_similar_codes, test_result


@app.cell
def __(mo):
    mo.md("## Step 7: Final Statistics")
    return


@app.cell
def __(conn, np):
    # Compute final statistics
    stats_df = conn.execute("""
        SELECT
            json_extract_string(json_data, '$.level') as level,
            COUNT(*) as code_count,
            AVG(CAST(json_extract(json_data, '$.relationship_stats.same_sector_count') AS DOUBLE)) as avg_same_sector,
            AVG(CAST(json_extract(json_data, '$.relationship_stats.cross_sector_count') AS DOUBLE)) as avg_cross_sector,
            AVG(CAST(json_extract(json_data, '$.relationship_stats.avg_similarity') AS DOUBLE)) as avg_similarity
        FROM naics_relationships
        GROUP BY level
        ORDER BY level
    """).fetchdf()

    print("=== Statistics by Level ===")
    stats_df
    return (stats_df,)


@app.cell
def __(conn):
    # Get database size
    _size_result = conn.execute("""
        SELECT
            SUM(LENGTH(json_data::VARCHAR)) as json_bytes
        FROM naics_relationships
    """).fetchone()
    _bytes = _size_result[0] if _size_result else 0
    _size_str = f"{_bytes / 1024 / 1024:.2f} MB" if _bytes else "N/A"
    print(f"Total JSON data size: {_size_str}")
    return


@app.cell
def __(conn):
    conn.close()
    print("\nDatabase connection closed. Relationships generated successfully!")
    return


@app.cell
def __(mo):
    mo.md(
        """
        ## Summary

        Generated **pre-computed semantic relationships** for all NAICS codes:

        - **Same-sector alternatives**: Codes in the same sector with high similarity
        - **Cross-sector alternatives**: Similar codes in different sectors (key insight!)
        - **Relationship statistics**: Count, max, and average similarity

        ### Output Table: `naics_relationships`

        | Column | Type | Description |
        |--------|------|-------------|
        | node_code | VARCHAR | NAICS code (primary key) |
        | json_data | JSON | Full relationship document |
        | created_at | TIMESTAMP | When computed |

        ### JSON Document Structure

        ```json
        {
          "node_code": "722511",
          "level": "national_industry",
          "title": "Full-Service Restaurants",
          "sector_code": "72",
          "hierarchy_path": ["72", "722", "7225", "72251", "722511"],
          "same_sector_alternatives": [...],
          "cross_sector_alternatives": [...],
          "relationship_stats": {
            "same_sector_count": 15,
            "cross_sector_count": 8,
            "max_similarity": 0.92,
            "avg_similarity": 0.81
          }
        }
        ```

        **Next step:** Use `get_similar_codes`, `get_cross_sector_alternatives`,
        and `get_relationship_stats` tools in the MCP server.
        """
    )
    return


if __name__ == "__main__":
    app.run()

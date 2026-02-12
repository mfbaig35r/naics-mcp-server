#!/usr/bin/env python3
"""
NAICS Embedding Generator

Generate vector embeddings for semantic search.
Run as script: python 05_generate_embeddings.py
Run interactive: marimo edit 05_generate_embeddings.py
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
        # NAICS 2022 Embedding Generator

        This notebook generates **384-dimensional vector embeddings** for semantic search
        using the `all-MiniLM-L6-v2` model from Sentence Transformers.

        Embeddings are created from `raw_embedding_text` which combines:
        - Title
        - Description (first 500 chars)
        - Top 10 index terms
        """
    )
    return


@app.cell
def __():
    import duckdb
    import numpy as np
    from pathlib import Path
    from sentence_transformers import SentenceTransformer
    import time
    return Path, SentenceTransformer, duckdb, np, time


@app.cell
def __(Path):
    # Configuration
    PROJECT_ROOT = Path(__file__).parent.parent if "__file__" in dir() else Path.cwd().parent
    DB_PATH = PROJECT_ROOT / "data" / "naics.duckdb"
    MODEL_NAME = "all-MiniLM-L6-v2"
    BATCH_SIZE = 32

    print(f"Database path: {DB_PATH}")
    print(f"Model: {MODEL_NAME}")
    return BATCH_SIZE, DB_PATH, MODEL_NAME, PROJECT_ROOT


@app.cell
def __(mo):
    mo.md("## Step 1: Load Embedding Model")
    return


@app.cell
def __(MODEL_NAME, SentenceTransformer, time):
    print(f"Loading model {MODEL_NAME}...")
    _start = time.time()

    model = SentenceTransformer(MODEL_NAME)

    _elapsed = time.time() - _start
    print(f"Model loaded in {_elapsed:.1f}s")
    print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return (model,)


@app.cell
def __(mo):
    mo.md("## Step 2: Load Codes for Embedding")
    return


@app.cell
def __(DB_PATH, duckdb):
    # Connect to database
    conn = duckdb.connect(str(DB_PATH))

    # Get codes that need embeddings
    df_codes = conn.execute("""
        SELECT node_code, title, raw_embedding_text
        FROM naics_nodes
        WHERE raw_embedding_text IS NOT NULL
        ORDER BY node_code
    """).fetchdf()

    print(f"Codes to embed: {len(df_codes)}")
    df_codes.head()
    return conn, df_codes


@app.cell
def __(df_codes):
    # Check embedding text quality
    df_codes["text_len"] = df_codes["raw_embedding_text"].str.len()
    print("Embedding text length statistics:")
    print(df_codes["text_len"].describe())

    # Sample embedding text
    _sample = df_codes[df_codes["node_code"] == "445110"].iloc[0]
    print(f"\nSample embedding text for 445110:")
    print(_sample["raw_embedding_text"][:300] + "...")
    return


@app.cell
def __(mo):
    mo.md("## Step 3: Generate Embeddings")
    return


@app.cell
def __(BATCH_SIZE, df_codes, model, np, time):
    # Generate embeddings in batches
    texts = df_codes["raw_embedding_text"].tolist()
    codes = df_codes["node_code"].tolist()

    print(f"Generating embeddings for {len(texts)} codes...")
    print(f"Batch size: {BATCH_SIZE}")

    _start = time.time()
    _all_embeddings = []

    for _i in range(0, len(texts), BATCH_SIZE):
        _batch_texts = texts[_i:_i+BATCH_SIZE]
        _batch_embeddings = model.encode(_batch_texts, show_progress_bar=False)
        _all_embeddings.extend(_batch_embeddings)

        if (_i + BATCH_SIZE) % 100 == 0 or _i + BATCH_SIZE >= len(texts):
            print(f"  Processed {min(_i + BATCH_SIZE, len(texts))}/{len(texts)}")

    embeddings = np.array(_all_embeddings)
    _elapsed = time.time() - _start

    print(f"\nGenerated {len(embeddings)} embeddings in {_elapsed:.1f}s")
    print(f"Embedding shape: {embeddings.shape}")
    return codes, embeddings, texts


@app.cell
def __(mo):
    mo.md("## Step 4: Store Embeddings in Database")
    return


@app.cell
def __(conn):
    # Create embeddings table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS naics_embeddings (
            node_code VARCHAR PRIMARY KEY,
            embedding FLOAT[384],
            embedding_text VARCHAR
        )
    """)

    # Clear existing
    conn.execute("DELETE FROM naics_embeddings")
    print("Embeddings table ready")
    return


@app.cell
def __(codes, conn, embeddings, texts):
    # Insert embeddings
    # DuckDB can handle array inserts
    print("Inserting embeddings...")

    for _idx, (_code, _emb, _text) in enumerate(zip(codes, embeddings, texts)):
        # Convert numpy array to list for DuckDB
        _emb_list = _emb.tolist()
        conn.execute(
            "INSERT INTO naics_embeddings (node_code, embedding, embedding_text) VALUES (?, ?, ?)",
            [_code, _emb_list, _text]
        )
        if (_idx + 1) % 200 == 0:
            print(f"  Inserted {_idx + 1}/{len(codes)}")

    print(f"Inserted all {len(codes)} embeddings")
    return


@app.cell
def __(conn):
    # Verify
    _count = conn.execute("SELECT COUNT(*) FROM naics_embeddings").fetchone()[0]
    print(f"Total embeddings in database: {_count}")

    # Check embedding dimensions
    _sample = conn.execute("""
        SELECT node_code, array_length(embedding, 1) as dim
        FROM naics_embeddings
        LIMIT 5
    """).fetchdf()
    _sample
    return


@app.cell
def __(mo):
    mo.md("## Step 5: Test Similarity Search")
    return


@app.cell
def __(conn, model, np):
    # Test: Find similar codes to a query
    _test_query = "grocery store selling food"
    _query_embedding = model.encode(_test_query)

    # Get all embeddings for comparison (in production, use vector index)
    _all_emb = conn.execute("""
        SELECT node_code, embedding
        FROM naics_embeddings
    """).fetchall()

    # Compute similarities
    _similarities = []
    for _code, _emb in _all_emb:
        _emb_array = np.array(_emb)
        # Cosine similarity
        _sim = np.dot(_query_embedding, _emb_array) / (
            np.linalg.norm(_query_embedding) * np.linalg.norm(_emb_array)
        )
        _similarities.append((_code, _sim))

    # Sort by similarity
    _similarities.sort(key=lambda x: x[1], reverse=True)

    print(f"Query: '{_test_query}'")
    print("\nTop 10 most similar codes:")
    for _code, _sim in _similarities[:10]:
        _title = conn.execute(f"SELECT title FROM naics_nodes WHERE node_code = '{_code}'").fetchone()[0]
        print(f"  [{_code}] {_title[:50]:50s} (similarity: {_sim:.3f})")
    return


@app.cell
def __(mo):
    mo.md("## Step 6: Final Statistics")
    return


@app.cell
def __(conn):
    # Coverage statistics
    stats = conn.execute("""
        SELECT
            n.level,
            COUNT(n.node_code) as total_codes,
            COUNT(e.node_code) as with_embeddings,
            ROUND(100.0 * COUNT(e.node_code) / COUNT(n.node_code), 1) as coverage_pct
        FROM naics_nodes n
        LEFT JOIN naics_embeddings e ON n.node_code = e.node_code
        GROUP BY n.level
        ORDER BY n.level
    """).fetchdf()

    print("Embedding coverage by level:")
    stats
    return (stats,)


@app.cell
def __(conn):
    conn.close()
    print("Database connection closed. Embeddings generated successfully!")
    return


@app.cell
def __(mo):
    mo.md(
        """
        ## Summary

        Generated **384-dimensional embeddings** for all NAICS codes using:
        - Model: `all-MiniLM-L6-v2`
        - Input: title + description + index terms
        - Storage: DuckDB array column

        The embeddings enable:
        - Semantic similarity search
        - "Find similar industries" functionality
        - Hybrid search (semantic + lexical)

        **Next step:** Run `99_explore_database.py` to interactively explore the data.
        """
    )
    return


if __name__ == "__main__":
    app.run()

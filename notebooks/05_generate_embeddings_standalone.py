#!/usr/bin/env python3
"""
NAICS Embedding Generator - Standalone Script

Generate vector embeddings for semantic search.
"""

import duckdb
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import time

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "naics.duckdb"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32

print(f"Database path: {DB_PATH}")
print(f"Model: {MODEL_NAME}")

# Load model
print(f"\nLoading model {MODEL_NAME}...")
start = time.time()
model = SentenceTransformer(MODEL_NAME)
elapsed = time.time() - start
print(f"Model loaded in {elapsed:.1f}s")
print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

# Connect to database
conn = duckdb.connect(str(DB_PATH))

# Get codes that need embeddings
df_codes = conn.execute("""
    SELECT node_code, title, raw_embedding_text
    FROM naics_nodes
    WHERE raw_embedding_text IS NOT NULL
    ORDER BY node_code
""").fetchdf()

print(f"\nCodes to embed: {len(df_codes)}")

# Check embedding text quality
df_codes["text_len"] = df_codes["raw_embedding_text"].str.len()
print("Embedding text length statistics:")
print(df_codes["text_len"].describe())

# Sample embedding text
sample = df_codes[df_codes["node_code"] == "445110"]
if len(sample) > 0:
    print(f"\nSample embedding text for 445110:")
    print(sample.iloc[0]["raw_embedding_text"][:300] + "...")

# Generate embeddings in batches
texts = df_codes["raw_embedding_text"].tolist()
codes = df_codes["node_code"].tolist()

print(f"\nGenerating embeddings for {len(texts)} codes...")
print(f"Batch size: {BATCH_SIZE}")

start = time.time()
all_embeddings = []

for i in range(0, len(texts), BATCH_SIZE):
    batch_texts = texts[i:i+BATCH_SIZE]
    batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
    all_embeddings.extend(batch_embeddings)

    if (i + BATCH_SIZE) % 100 == 0 or i + BATCH_SIZE >= len(texts):
        print(f"  Processed {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")

embeddings = np.array(all_embeddings)
elapsed = time.time() - start

print(f"\nGenerated {len(embeddings)} embeddings in {elapsed:.1f}s")
print(f"Embedding shape: {embeddings.shape}")

# Create embeddings table
conn.execute("DROP TABLE IF EXISTS naics_embeddings")
conn.execute("""
    CREATE TABLE naics_embeddings (
        node_code VARCHAR PRIMARY KEY,
        embedding FLOAT[384],
        embedding_text VARCHAR
    )
""")
print("\nEmbeddings table created")

# Insert embeddings
print("Inserting embeddings...")
for idx, (code, emb, text) in enumerate(zip(codes, embeddings, texts)):
    emb_list = emb.tolist()
    conn.execute(
        "INSERT INTO naics_embeddings (node_code, embedding, embedding_text) VALUES (?, ?, ?)",
        [code, emb_list, text]
    )
    if (idx + 1) % 500 == 0:
        print(f"  Inserted {idx + 1}/{len(codes)}")

print(f"Inserted all {len(codes)} embeddings")

# Verify
count = conn.execute("SELECT COUNT(*) FROM naics_embeddings").fetchone()[0]
print(f"\nTotal embeddings in database: {count}")

# Check embedding dimensions
sample_emb = conn.execute("""
    SELECT node_code, array_length(embedding, 1) as dim
    FROM naics_embeddings
    LIMIT 5
""").fetchdf()
print(sample_emb)

# Test similarity search
print("\n--- Testing Similarity Search ---")
test_query = "grocery store selling food"
query_embedding = model.encode(test_query)

all_emb = conn.execute("""
    SELECT node_code, embedding
    FROM naics_embeddings
""").fetchall()

similarities = []
for code, emb in all_emb:
    emb_array = np.array(emb)
    sim = np.dot(query_embedding, emb_array) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(emb_array)
    )
    similarities.append((code, sim))

similarities.sort(key=lambda x: x[1], reverse=True)

print(f"Query: '{test_query}'")
print("\nTop 10 most similar codes:")
for code, sim in similarities[:10]:
    title = conn.execute(f"SELECT title FROM naics_nodes WHERE node_code = '{code}'").fetchone()[0]
    print(f"  [{code}] {title[:50]:50s} (similarity: {sim:.3f})")

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

print("\nEmbedding coverage by level:")
print(stats)

conn.close()
print("\nDatabase connection closed. Embeddings generated successfully!")

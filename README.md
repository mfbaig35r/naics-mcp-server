# NAICS MCP Server

An intelligent industry classification service for NAICS 2022, built with the Model Context Protocol (MCP).

## Features

- **Semantic Search**: Natural language search for NAICS codes using embeddings
- **Hybrid Search**: Combines semantic understanding with exact term matching (70/30 split)
- **5-Level Hierarchy**: Navigate from Sector (2-digit) to National Industry (6-digit)
- **Cross-Reference Integration**: Critical exclusion checks for accurate classification
- **Index Term Search**: 20,398 official NAICS index terms
- **Classification Workbook**: Record and track classification decisions

## Quick Start

### Installation

```bash
pip install naics-mcp-server
```

Or install from source:

```bash
git clone https://github.com/your-org/naics-mcp-server.git
cd naics-mcp-server
pip install -e .
```

### Running the Server

```bash
# As MCP server
python -m naics_mcp_server

# CLI commands
naics-mcp search "retail grocery store"
naics-mcp hierarchy 445110
naics-mcp stats
```

### Configuration

Set environment variables:

```bash
export NAICS_DATABASE_PATH=data/naics.duckdb
export NAICS_EMBEDDING_MODEL=all-MiniLM-L6-v2
export NAICS_SEMANTIC_WEIGHT=0.7
export NAICS_MIN_CONFIDENCE=0.3
export NAICS_ENABLE_AUDIT=true
export DEBUG=false
```

## MCP Tools

### Search Tools (4)

| Tool | Purpose |
|------|---------|
| `search_naics_codes` | Hybrid semantic/lexical search with confidence scoring |
| `search_index_terms` | Search official 20,398-term index |
| `find_similar_industries` | Embedding similarity + cross-references |
| `classify_batch` | Batch classification for multiple descriptions |

### Hierarchy Tools (3)

| Tool | Purpose |
|------|---------|
| `get_code_hierarchy` | Full ancestor chain (Sector → National Industry) |
| `get_children` | Immediate children of a code |
| `get_siblings` | Codes at same level with same parent |

### Classification Tools (2)

| Tool | Purpose |
|------|---------|
| `classify_business` | Classify description → NAICS with reasoning |
| `get_cross_references` | Get exclusions/inclusions for a code |

### Analytics Tools (2)

| Tool | Purpose |
|------|---------|
| `get_sector_overview` | Summary of sector/subsector structure |
| `compare_codes` | Side-by-side comparison of multiple codes |

### Workbook Tools (4)

| Tool | Purpose |
|------|---------|
| `write_to_workbook` | Record classification decisions |
| `search_workbook` | Search past decisions |
| `get_workbook_entry` | Retrieve specific entry |
| `get_workbook_template` | Get form template |

## Database Schema

```sql
-- Primary codes (1,012 6-digit codes)
naics_nodes (
    node_code, level, title, description,
    sector_code, subsector_code, industry_group_code, naics_industry_code,
    raw_embedding_text, change_indicator, is_trilateral
)

-- Vector embeddings (384-dim)
naics_embeddings (node_code, embedding, embedding_text)

-- Official index terms (20,398 entries)
naics_index_terms (term_id, naics_code, index_term, term_normalized)

-- Cross-references (critical for classification)
naics_cross_references (ref_id, source_code, reference_type, reference_text, target_code)
```

## Confidence Scoring

```
overall = (
    0.40 * semantic_score +    # Core embedding match
    0.20 * lexical_score +     # Term overlap
    0.15 * index_term_match +  # Official index hit
    0.15 * specificity_pref +  # 6-digit > 2-digit
    0.10 * cross_ref_relevance # Classification guidance
)
```

## Data Sources

| Source File | Content |
|-------------|---------|
| `2-6 digit_2022_Codes.xlsx` | All NAICS codes + titles |
| `2022_NAICS_Descriptions.xlsx` | Full descriptions |
| `2022_NAICS_Index_File.xlsx` | 20,398 index terms |
| `2022_NAICS_Cross_References.xlsx` | Classification guidance |

## Performance Targets

| Metric | Target |
|--------|--------|
| Search latency | < 200ms |
| Batch (100 items) | < 10s |
| Server startup | < 3s (cached embeddings) |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black naics_mcp_server
ruff check naics_mcp_server
```

## License

MIT

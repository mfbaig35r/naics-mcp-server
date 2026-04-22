# NAICS MCP Server — Team Setup Guide

Self-contained NAICS 2022 industry classification server. No API keys, no external services — everything runs locally in a Docker container.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

## Quick Start

### 1. Pull the image

```bash
docker pull fbaig4/naics-mcp-server:latest
```

### 2. Verify it works

```bash
docker run --rm -i fbaig4/naics-mcp-server:latest
```

You should see log output ending with `NAICS MCP Server ready` and tool/code counts. Press `Ctrl+C` to stop.

### 3. Add to your MCP client

#### Claude Desktop / Claude Code

Add this to your MCP config (`~/.claude.json` or Claude Desktop settings):

```json
{
  "mcpServers": {
    "naics": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "fbaig4/naics-mcp-server:latest"]
    }
  }
}
```

#### Cursor

Add to `.cursor/mcp.json` in your project or `~/.cursor/mcp.json` globally:

```json
{
  "mcpServers": {
    "naics": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "fbaig4/naics-mcp-server:latest"]
    }
  }
}
```

### 4. Test it

Ask your AI assistant something like:

- "What NAICS code is for a coffee shop?"
- "Show me the hierarchy for code 445110"
- "Compare codes 541511 and 541512"

## What's included

| Component | Details |
|-----------|---------|
| NAICS codes | 2,125 codes across 5 levels |
| Index terms | 20,398 official terms |
| Cross-references | 4,601 exclusion/inclusion entries |
| Search | Hybrid semantic + lexical (all-MiniLM-L6-v2) |
| Database | DuckDB (embedded, no external DB) |
| Image size | ~2.1 GB (includes PyTorch CPU + model) |

## Available tools (25)

**Search**: `search_naics_codes`, `search_index_terms`, `find_similar_industries`, `get_similar_codes`, `get_cross_sector_alternatives`

**Hierarchy**: `get_code_hierarchy`, `get_children`, `get_siblings`

**Classification**: `classify_business`, `classify_batch`, `validate_classification`, `get_cross_references`

**Analytics**: `get_sector_overview`, `compare_codes`, `get_relationship_stats`

**Workbook**: `write_to_workbook`, `search_workbook`, `get_workbook_entry`, `get_workbook_template`

**Diagnostics**: `ping`, `check_readiness`, `get_server_health`, `get_metrics`, `get_shutdown_status`, `get_workflow_guide`

## Troubleshooting

**Image pull fails?**
Make sure Docker Desktop is running. If you get auth errors, try `docker login` first.

**Server slow to start?**
First run loads the sentence-transformers model (~15-20 seconds). Subsequent starts are faster with Docker's layer cache.

**MCP client can't connect?**
Ensure the `command` is `docker` (not `docker run`) and `args` includes `--rm` and `-i`. The `-i` flag is required for stdio transport.

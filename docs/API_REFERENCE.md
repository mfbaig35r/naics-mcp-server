# API Reference

Complete reference for all NAICS MCP Server tools, resources, and prompts.

## Table of Contents

- [Search Tools](#search-tools)
  - [search_naics_codes](#search_naics_codes)
  - [search_index_terms](#search_index_terms)
  - [find_similar_industries](#find_similar_industries)
  - [classify_batch](#classify_batch)
- [Hierarchy Tools](#hierarchy-tools)
  - [get_code_hierarchy](#get_code_hierarchy)
  - [get_children](#get_children)
  - [get_siblings](#get_siblings)
- [Classification Tools](#classification-tools)
  - [classify_business](#classify_business)
  - [get_cross_references](#get_cross_references)
  - [validate_classification](#validate_classification)
- [Analytics Tools](#analytics-tools)
  - [get_sector_overview](#get_sector_overview)
  - [compare_codes](#compare_codes)
- [Workbook Tools](#workbook-tools)
  - [write_to_workbook](#write_to_workbook)
  - [search_workbook](#search_workbook)
  - [get_workbook_entry](#get_workbook_entry)
  - [get_workbook_template](#get_workbook_template)
- [Diagnostic Tools](#diagnostic-tools)
  - [ping](#ping)
  - [check_readiness](#check_readiness)
  - [get_server_health](#get_server_health)
  - [get_metrics](#get_metrics)
  - [get_shutdown_status](#get_shutdown_status)
  - [get_workflow_guide](#get_workflow_guide)
- [Resources](#resources)
- [Prompts](#prompts)

---

## Search Tools

### search_naics_codes

Search for NAICS codes using natural language descriptions.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | Natural language description of the business or activity |
| `strategy` | string | No | `"hybrid"` | Search strategy: `hybrid`, `semantic`, or `lexical` |
| `limit` | integer | No | `10` | Maximum results (1-50) |
| `min_confidence` | float | No | `0.3` | Minimum confidence threshold (0.0-1.0) |
| `include_cross_refs` | boolean | No | `true` | Include cross-reference exclusion checks |

**Search Strategies:**

| Strategy | Description | Best For |
|----------|-------------|----------|
| `hybrid` | Balances semantic meaning with exact term matching | General use (recommended) |
| `semantic` | Focuses on conceptual similarity | Vague descriptions, related industries |
| `lexical` | Prioritizes exact term matching | Known terminology, specific terms |

**Response:**

```json
{
  "query": "retail grocery store",
  "results": [
    {
      "code": "445110",
      "title": "Supermarkets and Other Grocery Retailers",
      "description": "This industry comprises establishments...",
      "level": "national_industry",
      "confidence": 0.92,
      "explanation": "Strong semantic match (0.89), lexical boost (0.85), 6-digit specificity bonus",
      "hierarchy": ["44-45", "445", "4451", "44511", "445110"],
      "matched_index_terms": ["grocery stores", "supermarkets"],
      "exclusion_warnings": []
    }
  ],
  "expanded": false,
  "strategy_used": "hybrid",
  "total_found": 5,
  "search_time_ms": 45,
  "guidance": ["Consider checking 445120 for convenience stores"]
}
```

**Example:**

```python
# Search for a business description
result = await search_naics_codes({
    "query": "company that manufactures dog food",
    "strategy": "hybrid",
    "limit": 5,
    "min_confidence": 0.5
})
```

---

### search_index_terms

Search the official NAICS index terms (20,398 official mappings).

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `search_text` | string | Yes | - | Text to search in index terms |
| `limit` | integer | No | `20` | Maximum results |

**Response:**

```json
{
  "search_text": "dog grooming",
  "matches": [
    {"index_term": "Dog grooming services", "naics_code": "812910"},
    {"index_term": "Pet grooming services", "naics_code": "812910"}
  ],
  "total_found": 2
}
```

**Example:**

```python
# Find official index terms for software
result = await search_index_terms({
    "search_text": "software publishing",
    "limit": 10
})
```

---

### find_similar_industries

Find NAICS codes similar to a given code using semantic similarity.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `naics_code` | string | Yes | - | NAICS code to find similar codes for |
| `limit` | integer | No | `5` | Maximum results (1-20) |
| `min_similarity` | float | No | `0.7` | Minimum similarity threshold (0.0-1.0) |

**Response:**

```json
{
  "original_code": "445110",
  "original_title": "Supermarkets and Other Grocery Retailers",
  "similar_codes": [
    {
      "code": "445120",
      "title": "Convenience Retailers",
      "similarity": 0.87,
      "level": "national_industry"
    },
    {
      "code": "445230",
      "title": "Fruit and Vegetable Retailers",
      "similarity": 0.81,
      "level": "national_industry"
    }
  ]
}
```

---

### classify_batch

Classify multiple business descriptions efficiently.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `descriptions` | string[] | Yes | - | List of business descriptions (max 100) |
| `include_confidence` | boolean | No | `true` | Include confidence scores |

**Constraints:**

- Maximum 100 descriptions per batch
- Each description must be 10-5,000 characters

**Response:**

```json
{
  "classifications": [
    {
      "description": "retail grocery store",
      "naics_code": "445110",
      "naics_title": "Supermarkets and Other Grocery Retailers",
      "level": "national_industry",
      "confidence": 0.92,
      "explanation": "Strong match based on..."
    },
    {
      "description": "dog food manufacturing",
      "naics_code": "311111",
      "naics_title": "Dog and Cat Food Manufacturing",
      "level": "national_industry",
      "confidence": 0.95
    }
  ],
  "total_processed": 2,
  "successfully_classified": 2
}
```

---

## Hierarchy Tools

### get_code_hierarchy

Get the complete hierarchical path for a NAICS code.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `naics_code` | string | Yes | NAICS code (2-6 digits) |

**Response:**

```json
{
  "naics_code": "445110",
  "hierarchy": [
    {
      "level": "sector",
      "code": "44-45",
      "title": "Retail Trade",
      "description": "The Retail Trade sector comprises establishments..."
    },
    {
      "level": "subsector",
      "code": "445",
      "title": "Food and Beverage Retailers",
      "description": "Industries in the Food and Beverage Retailers..."
    },
    {
      "level": "industry_group",
      "code": "4451",
      "title": "Grocery and Convenience Retailers",
      "description": "This industry group comprises establishments..."
    },
    {
      "level": "naics_industry",
      "code": "44511",
      "title": "Supermarkets and Other Grocery Retailers (except Convenience Retailers)",
      "description": "..."
    },
    {
      "level": "national_industry",
      "code": "445110",
      "title": "Supermarkets and Other Grocery Retailers",
      "description": "This industry comprises establishments..."
    }
  ]
}
```

---

### get_children

Get immediate children of a NAICS code in the hierarchy.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `naics_code` | string | Yes | Parent NAICS code |

**Response:**

```json
{
  "parent_code": "445",
  "children": [
    {"code": "4451", "title": "Grocery and Convenience Retailers", "level": "industry_group"},
    {"code": "4452", "title": "Specialty Food Retailers", "level": "industry_group"},
    {"code": "4453", "title": "Beer, Wine, and Liquor Retailers", "level": "industry_group"}
  ],
  "count": 3
}
```

---

### get_siblings

Get sibling codes at the same hierarchical level.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `naics_code` | string | Yes | - | NAICS code |
| `limit` | integer | No | `10` | Maximum results |

**Response:**

```json
{
  "code": "445110",
  "title": "Supermarkets and Other Grocery Retailers",
  "level": "national_industry",
  "siblings": [
    {"code": "445120", "title": "Convenience Retailers"},
    {"code": "445131", "title": "Milk Retailers"},
    {"code": "445132", "title": "Dairy Product Retailers (except Milk Retailers)"}
  ]
}
```

---

## Classification Tools

### classify_business

Classify a business description with detailed reasoning.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `description` | string | Yes | - | Business or activity description |
| `include_reasoning` | boolean | No | `true` | Include detailed reasoning |
| `check_cross_refs` | boolean | No | `true` | Check cross-references for exclusions |

**Response:**

```json
{
  "input": "Company that manufactures organic dog treats...",
  "recommendation": {
    "code": "311111",
    "title": "Dog and Cat Food Manufacturing",
    "level": "national_industry",
    "confidence": 0.91,
    "reasoning": "Strong match for pet food manufacturing..."
  },
  "alternatives": [
    {
      "code": "311119",
      "title": "Other Animal Food Manufacturing",
      "confidence": 0.72,
      "why_not_primary": "Less specific than 311111 for dog treats"
    }
  ],
  "hierarchy": ["31-33", "311", "3111", "31111", "311111"],
  "index_term_matches": ["dog food manufacturing", "pet food"],
  "cross_reference_checks": {
    "exclusions": [],
    "inclusions": ["Dog and cat food, including treats, manufacturing"]
  },
  "confidence_breakdown": {
    "semantic": 0.89,
    "lexical": 0.85,
    "index_term": 0.15,
    "specificity": 0.10
  }
}
```

---

### get_cross_references

Get cross-references (exclusions and inclusions) for a NAICS code.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `naics_code` | string | Yes | NAICS code to check |

**Response:**

```json
{
  "naics_code": "445110",
  "title": "Supermarkets and Other Grocery Retailers",
  "cross_references": [
    {
      "type": "exclusion",
      "text": "Convenience stores are classified in 445120",
      "target_code": "445120",
      "excluded_activity": "Convenience stores"
    },
    {
      "type": "inclusion",
      "text": "Grocery stores selling primarily food products",
      "target_code": null
    }
  ],
  "total_references": 5
}
```

---

### validate_classification

Validate if a NAICS code is appropriate for a business description.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `description` | string | Yes | Business description |
| `proposed_code` | string | Yes | NAICS code to validate |

**Response:**

```json
{
  "proposed_code": "445110",
  "proposed_title": "Supermarkets and Other Grocery Retailers",
  "description": "Retail store selling groceries...",
  "validation": {
    "is_valid": true,
    "confidence": 0.89,
    "reasoning": "Strong match - description aligns with industry definition"
  },
  "better_alternatives": [],
  "warnings": []
}
```

---

## Analytics Tools

### get_sector_overview

Get summary of sector/subsector structure.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `sector_code` | string | No | - | Specific sector code (optional) |

**Response:**

```json
{
  "sectors": [
    {
      "code": "44-45",
      "title": "Retail Trade",
      "subsector_count": 12,
      "total_codes": 185
    }
  ],
  "total_sectors": 20,
  "total_codes": 2125
}
```

---

### compare_codes

Side-by-side comparison of multiple NAICS codes.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `codes` | string[] | Yes | List of NAICS codes to compare (2-5 codes) |

**Response:**

```json
{
  "codes": [
    {
      "code": "445110",
      "title": "Supermarkets and Other Grocery Retailers",
      "level": "national_industry",
      "sector": "Retail Trade",
      "description": "..."
    },
    {
      "code": "445120",
      "title": "Convenience Retailers",
      "level": "national_industry",
      "sector": "Retail Trade",
      "description": "..."
    }
  ],
  "common_ancestors": ["44-45", "445", "4451"],
  "key_differences": [
    "445110 focuses on grocery stores with wide selection",
    "445120 emphasizes convenience and extended hours"
  ]
}
```

---

## Workbook Tools

### write_to_workbook

Record a classification decision in the workbook.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `business_description` | string | Yes | Business description |
| `selected_code` | string | Yes | Selected NAICS code |
| `selected_title` | string | Yes | Title of selected code |
| `confidence` | float | No | Confidence score (0-1) |
| `reasoning` | string | No | Classification reasoning |
| `alternatives_considered` | string[] | No | Alternative codes considered |
| `cross_refs_checked` | boolean | No | Whether cross-refs were checked |
| `form_type` | string | No | Form type: `basic`, `detailed`, `compliance` |
| `metadata` | object | No | Additional metadata |

**Response:**

```json
{
  "success": true,
  "entry_id": "wb_2024_001",
  "message": "Classification recorded successfully",
  "timestamp": "2024-01-15T10:30:45Z"
}
```

---

### search_workbook

Search past classification decisions.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | No | - | Search in descriptions |
| `naics_code` | string | No | - | Filter by code |
| `limit` | integer | No | `20` | Maximum results |

**Response:**

```json
{
  "entries": [
    {
      "entry_id": "wb_2024_001",
      "business_description": "...",
      "selected_code": "445110",
      "confidence": 0.92,
      "created_at": "2024-01-15T10:30:45Z"
    }
  ],
  "total_found": 1
}
```

---

### get_workbook_entry

Retrieve a specific workbook entry.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `entry_id` | string | Yes | Workbook entry ID |

**Response:**

```json
{
  "entry_id": "wb_2024_001",
  "business_description": "Retail grocery store...",
  "selected_code": "445110",
  "selected_title": "Supermarkets and Other Grocery Retailers",
  "confidence": 0.92,
  "reasoning": "Strong match based on...",
  "alternatives_considered": ["445120"],
  "cross_refs_checked": true,
  "form_type": "detailed",
  "metadata": {},
  "created_at": "2024-01-15T10:30:45Z"
}
```

---

### get_workbook_template

Get a form template for structured classification input.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `form_type` | string | No | `"basic"` | Template type: `basic`, `detailed`, `compliance` |

**Response:**

```json
{
  "form_type": "detailed",
  "fields": [
    {"name": "business_description", "type": "text", "required": true},
    {"name": "primary_activity", "type": "text", "required": true},
    {"name": "products_services", "type": "text", "required": false},
    {"name": "customer_type", "type": "select", "options": ["B2B", "B2C", "Both"]}
  ],
  "instructions": "Complete all required fields..."
}
```

---

## Diagnostic Tools

### ping

Simple liveness check - confirms server process is alive.

**Parameters:** None

**Response:**

```json
{
  "status": "alive",
  "timestamp": "2024-01-15T10:30:45.123456+00:00"
}
```

---

### check_readiness

Check if server is ready to handle requests.

**Parameters:** None

**Response:**

```json
{
  "status": "ready",
  "uptime_seconds": 120.5
}
```

Or if not ready:

```json
{
  "status": "not_ready",
  "uptime_seconds": 5.2
}
```

---

### get_server_health

Comprehensive health check with detailed component diagnostics.

**Parameters:** None

**Response:**

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "uptime_seconds": 3600,
  "components": {
    "database": {
      "status": "ready",
      "total_codes": 2125,
      "total_index_terms": 20398
    },
    "embedder": {
      "status": "ready",
      "model": "all-MiniLM-L6-v2",
      "dimension": 384
    },
    "search_engine": {
      "status": "ready",
      "embedding_cache_hits": 1523,
      "search_cache_hits": 892
    },
    "embeddings": {
      "status": "ready",
      "coverage_percent": 100.0
    },
    "cross_references": {
      "status": "ready",
      "total_references": 4500
    },
    "workbook": {
      "status": "ready",
      "entries": 47
    }
  }
}
```

**Status Values:**

| Status | Meaning |
|--------|---------|
| `healthy` | All components ready |
| `degraded` | Some components partial, but operational |
| `unhealthy` | Critical components failed |

---

### get_metrics

Get Prometheus metrics for monitoring.

**Parameters:** None

**Response:**

```json
{
  "format": "prometheus",
  "metrics_count": 12,
  "raw_metrics": "# HELP naics_requests_total Total requests\n..."
}
```

---

### get_shutdown_status

Get current shutdown manager status.

**Parameters:** None

**Response:**

```json
{
  "state": "running",
  "in_flight_requests": 2,
  "registered_hooks": ["http_server", "database", "embedder"],
  "config": {
    "timeout_seconds": 30,
    "grace_period_seconds": 5
  }
}
```

**Shutdown States:**

| State | Description |
|-------|-------------|
| `running` | Normal operation |
| `draining` | Waiting for in-flight requests to complete |
| `stopping` | Executing shutdown hooks |
| `stopped` | Shutdown complete |

---

### get_workflow_guide

Get recommended classification workflows for different scenarios.

**Parameters:** None

**Response:**

```json
{
  "workflows": {
    "simple_classification": {
      "description": "Basic single-business classification",
      "steps": [
        "Use classify_business with the description",
        "Review confidence and alternatives",
        "Check cross-references if confidence < 0.8"
      ]
    },
    "batch_processing": {
      "description": "Classify multiple businesses",
      "steps": [
        "Use classify_batch with descriptions",
        "Review low-confidence results",
        "Use write_to_workbook for audit trail"
      ]
    }
  }
}
```

---

## Resources

### naics://statistics

Database and server statistics.

**Response:**

```json
{
  "database": {
    "total_codes": 2125,
    "codes_by_level": {
      "sector": 20,
      "subsector": 99,
      "industry_group": 312,
      "naics_industry": 713,
      "national_industry": 981
    },
    "total_index_terms": 20398,
    "total_cross_references": 4500
  },
  "embedding_cache": {
    "size": 2125,
    "hits": 15230,
    "misses": 125
  },
  "search_cache": {
    "size": 500,
    "hits": 8920,
    "misses": 1200
  }
}
```

### naics://recent_searches

Recent search queries for monitoring.

**Response:**

```json
[
  {
    "query": "retail grocery store",
    "strategy": "hybrid",
    "results_count": 5,
    "top_confidence": 0.92,
    "timestamp": "2024-01-15T10:30:45Z"
  }
]
```

---

## Prompts

### classification_assistant_prompt

Template for NAICS classification assistance. Returns a system prompt for an LLM to act as a NAICS classification expert.

---

## Error Handling

All tools return errors in a consistent format:

```json
{
  "error": "Error message",
  "error_category": "validation|not_found|server_error|rate_limited",
  "details": {}
}
```

**Error Categories:**

| Category | Description | HTTP Equivalent |
|----------|-------------|-----------------|
| `validation` | Invalid input parameters | 400 |
| `not_found` | Requested resource not found | 404 |
| `server_error` | Internal server error | 500 |
| `rate_limited` | Too many requests | 429 |
| `timeout` | Operation timed out | 504 |

---

## Rate Limiting

When rate limiting is enabled:

| Tool Category | Limit |
|---------------|-------|
| Search tools | 100/minute |
| Batch tools | 10/minute |
| Diagnostic tools | 60/minute |

Rate limit errors return:

```json
{
  "error": "Rate limit exceeded",
  "error_category": "rate_limited",
  "retry_after_seconds": 30
}
```

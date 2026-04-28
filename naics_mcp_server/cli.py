#!/usr/bin/env python3
"""
Command-line interface for NAICS MCP Server.

Provides commands for:
- Running the MCP server
- Loading/building the database
- Testing search functionality
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False):
    """Configure logging for CLI operations."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def cmd_serve(args):
    """Run the MCP server."""
    setup_logging(args.debug)
    logger.info("Starting NAICS MCP Server...")

    from .server import mcp

    mcp.run()


def cmd_init(args):
    """Initialize the database (create schema, optionally load data)."""
    setup_logging(args.debug)

    from .config import SearchConfig
    from .core.database import NAICSDatabase

    config = SearchConfig.from_env()
    db_path = Path(args.database) if args.database else config.database_path

    logger.info(f"Initializing database at {db_path}")

    db = NAICSDatabase(db_path)
    db.connect()

    stats = asyncio.run(db.get_statistics())
    logger.info(f"Database initialized. Statistics: {stats}")

    db.disconnect()
    logger.info("Done!")


def cmd_embeddings(args):
    """Generate or rebuild embeddings."""
    setup_logging(args.debug)

    from .config import SearchConfig
    from .core.database import NAICSDatabase
    from .core.embeddings import TextEmbedder
    from .core.search_engine import NAICSSearchEngine

    config = SearchConfig.from_env()
    db_path = Path(args.database) if args.database else config.database_path

    logger.info(f"Loading database from {db_path}")

    db = NAICSDatabase(db_path)
    db.connect()

    cache_dir = Path.home() / ".cache" / "naics-mcp-server" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)

    embedder = TextEmbedder(model_name=config.embedding_model, cache_dir=cache_dir)
    embedder.load_model()

    search_engine = NAICSSearchEngine(db, embedder, config)

    logger.info("Initializing embeddings...")
    result = asyncio.run(search_engine.initialize_embeddings(force_rebuild=args.rebuild))
    logger.info(f"Result: {result}")

    db.disconnect()
    logger.info("Done!")


def cmd_search(args):
    """Perform a test search."""
    setup_logging(args.debug)

    from .config import SearchConfig
    from .core.database import NAICSDatabase
    from .core.embeddings import TextEmbedder
    from .core.search_engine import NAICSSearchEngine
    from .models.search_models import SearchStrategy

    config = SearchConfig.from_env()
    db_path = Path(args.database) if args.database else config.database_path

    db = NAICSDatabase(db_path)
    db.connect()

    cache_dir = Path.home() / ".cache" / "naics-mcp-server" / "models"
    embedder = TextEmbedder(model_name=config.embedding_model, cache_dir=cache_dir)
    embedder.load_model()

    search_engine = NAICSSearchEngine(db, embedder, config)
    asyncio.run(search_engine.initialize_embeddings())

    strategy_map = {
        "hybrid": SearchStrategy.HYBRID,
        "semantic": SearchStrategy.SEMANTIC,
        "lexical": SearchStrategy.LEXICAL,
    }
    strategy = strategy_map.get(args.strategy, SearchStrategy.HYBRID)

    logger.info(f"Searching for: '{args.query}'")
    results = asyncio.run(
        search_engine.search(query=args.query, strategy=strategy, limit=args.limit)
    )

    print(f"\nResults for '{args.query}':")
    print(f"Strategy: {strategy.value}")
    print(f"Search time: {results.query_metadata.processing_time_ms}ms")
    print(f"Expanded: {results.query_metadata.was_expanded}")
    print("-" * 60)

    for match in results.matches:
        print(f"\n{match.rank}. [{match.code.node_code}] {match.code.title}")
        print(f"   Level: {match.code.level.value}")
        print(f"   Confidence: {match.confidence.overall:.1%}")
        print(f"   {match.confidence.to_explanation()}")
        if match.matched_index_terms:
            print(f"   Index terms: {', '.join(match.matched_index_terms[:3])}")
        if match.exclusion_warnings:
            print(f"   ⚠️  WARNINGS: {'; '.join(match.exclusion_warnings)}")

    db.disconnect()


def cmd_stats(args):
    """Show database statistics."""
    setup_logging(args.debug)

    from .config import SearchConfig
    from .core.database import NAICSDatabase

    config = SearchConfig.from_env()
    db_path = Path(args.database) if args.database else config.database_path

    db = NAICSDatabase(db_path)
    db.connect()

    stats = asyncio.run(db.get_statistics())

    print("\nNAICS Database Statistics")
    print("=" * 40)

    print("\nCodes by Level:")
    for level, count in stats.get("counts_by_level", {}).items():
        print(f"  {level}: {count:,}")

    print(f"\nTotal codes: {stats.get('total_codes', 0):,}")
    print(f"Index terms: {stats.get('total_index_terms', 0):,}")
    print(f"Cross-references: {stats.get('total_cross_references', 0):,}")

    embedding_stats = stats.get("embedding_coverage", {})
    if embedding_stats:
        print(f"\nEmbedding coverage: {embedding_stats.get('coverage_percent', 0):.1f}%")

    db.disconnect()


def cmd_hierarchy(args):
    """Show hierarchy for a NAICS code."""
    setup_logging(args.debug)

    from .config import SearchConfig
    from .core.database import NAICSDatabase

    config = SearchConfig.from_env()
    db_path = Path(args.database) if args.database else config.database_path

    db = NAICSDatabase(db_path)
    db.connect()

    hierarchy = asyncio.run(db.get_hierarchy(args.code))

    if not hierarchy:
        print(f"Code {args.code} not found")
    else:
        print(f"\nHierarchy for {args.code}:")
        print("=" * 60)
        for code in hierarchy:
            indent = "  " * (len(code.node_code) - 2)
            print(f"{indent}[{code.node_code}] {code.title}")

    db.disconnect()


def cmd_classify_batch(args):
    """Classify business descriptions from a CSV file in batch."""
    setup_logging(args.debug)

    import csv
    import time

    from .config import SearchConfig
    from .core.database import NAICSDatabase
    from .core.embeddings import TextEmbedder
    from .core.search_engine import NAICSSearchEngine
    from .models.search_models import SearchStrategy

    config = SearchConfig.from_env()
    db_path = Path(args.database) if args.database else config.database_path

    # Read input CSV
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    logger.info(f"Reading {input_path}")
    with open(input_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if args.column not in reader.fieldnames:
            logger.error(
                f"Column '{args.column}' not found. Available: {', '.join(reader.fieldnames)}"
            )
            sys.exit(1)
        rows = list(reader)

    total = len(rows)
    logger.info(f"Loaded {total:,} rows, classifying column '{args.column}'")

    # Initialize search engine
    db = NAICSDatabase(db_path)
    db.connect()

    cache_dir = Path.home() / ".cache" / "naics-mcp-server" / "models"
    embedder = TextEmbedder(model_name=config.embedding_model, cache_dir=cache_dir)
    embedder.load_model()

    search_engine = NAICSSearchEngine(db, embedder, config)
    asyncio.run(search_engine.initialize_embeddings())

    strategy_map = {
        "hybrid": SearchStrategy.HYBRID,
        "semantic": SearchStrategy.SEMANTIC,
        "lexical": SearchStrategy.LEXICAL,
    }
    strategy = strategy_map.get(args.strategy, SearchStrategy.HYBRID)
    top_n = args.top_n

    # Classify each row
    output_path = Path(args.output) if args.output else input_path.with_name(
        f"{input_path.stem}_classified{input_path.suffix}"
    )

    # Build output fieldnames: original columns + classification columns
    original_fields = list(rows[0].keys()) if rows else []
    classification_fields = []
    for i in range(1, top_n + 1):
        suffix = f"_{i}" if top_n > 1 else ""
        classification_fields.extend([
            f"naics_code{suffix}",
            f"naics_title{suffix}",
            f"naics_level{suffix}",
            f"confidence{suffix}",
        ])
    output_fields = original_fields + classification_fields

    start_time = time.time()
    classified = 0
    failed = 0

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()

        for i, row in enumerate(rows):
            description = (row.get(args.column) or "").strip()
            out_row = dict(row)

            if not description:
                # Fill classification columns with empty values
                for field_name in classification_fields:
                    out_row[field_name] = ""
                writer.writerow(out_row)
                failed += 1
            else:
                try:
                    results = asyncio.run(
                        search_engine.search(
                            query=description,
                            strategy=strategy,
                            limit=top_n,
                            min_confidence=0.2,
                        )
                    )

                    for j in range(top_n):
                        suffix = f"_{j + 1}" if top_n > 1 else ""
                        if j < len(results.matches):
                            match = results.matches[j]
                            out_row[f"naics_code{suffix}"] = match.code.node_code
                            out_row[f"naics_title{suffix}"] = match.code.title
                            out_row[f"naics_level{suffix}"] = match.code.level.value
                            out_row[f"confidence{suffix}"] = f"{match.confidence.overall:.4f}"
                        else:
                            out_row[f"naics_code{suffix}"] = ""
                            out_row[f"naics_title{suffix}"] = ""
                            out_row[f"naics_level{suffix}"] = ""
                            out_row[f"confidence{suffix}"] = ""

                    writer.writerow(out_row)
                    classified += 1

                except Exception as e:
                    logger.warning(f"Row {i + 1} failed: {e}")
                    for field_name in classification_fields:
                        out_row[field_name] = ""
                    writer.writerow(out_row)
                    failed += 1

            # Progress reporting
            processed = i + 1
            if processed % 500 == 0 or processed == total:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = (total - processed) / rate if rate > 0 else 0
                print(
                    f"  [{processed:,}/{total:,}] "
                    f"{processed / total:.0%} complete | "
                    f"{rate:.0f} rows/sec | "
                    f"~{remaining:.0f}s remaining",
                    flush=True,
                )

    elapsed = time.time() - start_time
    db.disconnect()

    print(f"\nDone in {elapsed:.1f}s ({total / elapsed:.0f} rows/sec)")
    print(f"  Classified: {classified:,}")
    print(f"  Failed/empty: {failed:,}")
    print(f"  Output: {output_path}")


def cmd_metrics_server(args):
    """Run the Prometheus metrics HTTP server."""
    setup_logging(args.debug)
    logger.info(f"Starting metrics server on {args.host}:{args.port}")

    from .observability.metrics_server import run_metrics_server

    run_metrics_server(host=args.host, port=args.port)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="NAICS MCP Server CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  naics-mcp serve                                  Run the MCP server
  naics-mcp search "retail grocery"                Search for NAICS codes
  naics-mcp hierarchy 445110                       Show hierarchy for a code
  naics-mcp stats                                  Show database statistics
  naics-mcp classify-batch suppliers.csv           Classify from CSV
  naics-mcp classify-batch in.csv --column name    Use a different column
  naics-mcp classify-batch in.csv --top-n 3        Get top 3 matches per row
        """,
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    parser.add_argument(
        "--database", type=str, help="Path to database file (overrides environment)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Run the MCP server")
    serve_parser.set_defaults(func=cmd_serve)

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize the database")
    init_parser.set_defaults(func=cmd_init)

    # embeddings command
    embed_parser = subparsers.add_parser("embeddings", help="Generate embeddings")
    embed_parser.add_argument(
        "--rebuild", action="store_true", help="Force rebuild of all embeddings"
    )
    embed_parser.set_defaults(func=cmd_embeddings)

    # search command
    search_parser = subparsers.add_parser("search", help="Perform a test search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--strategy",
        type=str,
        default="hybrid",
        choices=["hybrid", "semantic", "lexical"],
        help="Search strategy (default: hybrid)",
    )
    search_parser.add_argument(
        "--limit", type=int, default=10, help="Maximum results (default: 10)"
    )
    search_parser.set_defaults(func=cmd_search)

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.set_defaults(func=cmd_stats)

    # hierarchy command
    hierarchy_parser = subparsers.add_parser("hierarchy", help="Show code hierarchy")
    hierarchy_parser.add_argument("code", type=str, help="NAICS code")
    hierarchy_parser.set_defaults(func=cmd_hierarchy)

    # classify-batch command
    batch_parser = subparsers.add_parser(
        "classify-batch", help="Classify business descriptions from a CSV file"
    )
    batch_parser.add_argument("input_file", type=str, help="Input CSV file path")
    batch_parser.add_argument(
        "--column",
        type=str,
        default="description",
        help="Column containing descriptions (default: description)",
    )
    batch_parser.add_argument(
        "--output", type=str, help="Output CSV path (default: <input>_classified.csv)"
    )
    batch_parser.add_argument(
        "--strategy",
        type=str,
        default="hybrid",
        choices=["hybrid", "semantic", "lexical"],
        help="Search strategy (default: hybrid)",
    )
    batch_parser.add_argument(
        "--top-n",
        type=int,
        default=1,
        help="Number of top matches per row (default: 1)",
    )
    batch_parser.set_defaults(func=cmd_classify_batch)

    # metrics-server command
    metrics_parser = subparsers.add_parser(
        "metrics-server", help="Run Prometheus metrics HTTP server"
    )
    metrics_parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    metrics_parser.add_argument(
        "--port", type=int, default=9090, help="Port to listen on (default: 9090)"
    )
    metrics_parser.set_defaults(func=cmd_metrics_server)

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Run the command
    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

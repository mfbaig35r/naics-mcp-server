#!/usr/bin/env python3
"""
NAICS MCP Server entry point.

Run with: python -m naics_mcp_server [command] [options]

Without a command, starts the MCP server.
"""

import sys


def main():
    """Main entry point."""
    # If no arguments or just flags, run the server
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1].startswith("--")):
        from .server import main as server_main

        server_main()
    else:
        # Run CLI for other commands
        from .cli import main as cli_main

        cli_main()


if __name__ == "__main__":
    main()

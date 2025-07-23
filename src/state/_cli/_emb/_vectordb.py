import argparse as ap
import logging
import json
import yaml
from typing import Dict, Any


def add_arguments_vectordb(parser: ap.ArgumentParser):
    """Add arguments for state embedding vectordb CLI."""
    parser.add_argument("--lancedb", required=True, help="Path to existing LanceDB database")
    parser.add_argument("--format", choices=["json", "yaml", "table"], default="table", 
                       help="Output format for database summary")


def run_emb_vectordb(args: ap.ArgumentParser):
    """
    Get summary statistics about a LanceDB vector database.
    """
    logging.basicConfig(level=getattr(logging, args.log_level, logging.INFO))
    logger = logging.getLogger(__name__)
    
    from ...emb.vectordb import StateVectorDB
    
    # Connect to database
    logger.info(f"Connecting to database at {args.lancedb}")
    vector_db = StateVectorDB(args.lancedb)
    
    # Get database summary
    summary = vector_db.get_database_summary()
    
    # Output in requested format
    if args.format == "json":
        print(json.dumps(summary, indent=2))
    elif args.format == "yaml":
        print(yaml.dump(summary, default_flow_style=False))
    elif args.format == "table":
        _print_table_summary(summary)
    
    logger.info("Database summary completed successfully!")


def _print_table_summary(summary: Dict[str, Any]) -> None:
    """Print database summary in a nice table format."""
    if not summary["table_exists"]:
        print("❌ Database table does not exist")
        return
    
    if summary["num_cells"] == 0:
        print("⚠️  Database table exists but is empty")
        return
    
    # Print header
    print("=" * 60)
    print("📊 STATE VECTOR DATABASE SUMMARY")
    print("=" * 60)
    
    # Basic stats
    print(f"🔢 Total cells:        {summary['num_cells']:,}")
    print(f"📦 Total datasets:     {summary['num_datasets']}")
    print(f"🔑 Embedding keys:     {summary['num_embedding_keys']}")
    print(f"📐 Embedding dimension: {summary['embedding_dim']}")
    print()
    
    # Datasets breakdown
    if summary["datasets"]:
        print("📂 DATASETS:")
        for dataset in summary["datasets"]:
            cell_count = summary["cells_per_dataset"].get(dataset, 0)
            print(f"   • {dataset}: {cell_count:,} cells")
        print()
    
    # Embedding keys
    if summary["embedding_keys"]:
        print("🗝️  EMBEDDING KEYS:")
        for key in summary["embedding_keys"]:
            print(f"   • {key}")
        print()
    
    print("=" * 60) 
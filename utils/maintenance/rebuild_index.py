#!/usr/bin/env python3
# Script to rebuild the search index

import os
import sys
import logging
import shutil

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import resolve_path, retrieval_config
from core.document_processor import DocumentProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def rebuild_index():
    """
    Rebuild the search index from scratch
    """
    try:
        # Resolve paths
        index_dir = resolve_path(retrieval_config.index_path, create_dir=True)
        current_index_dir = resolve_path("current_index", create_dir=True)

        # Create backup of existing index if it exists
        if os.path.exists(index_dir) and os.listdir(index_dir):
            backup_dir = f"{index_dir}_backup_{int(time.time())}"
            logger.info(f"Creating backup of existing index at {backup_dir}")
            shutil.copytree(index_dir, backup_dir)

        # Clear existing index
        if os.path.exists(index_dir):
            for file in os.listdir(index_dir):
                file_path = os.path.join(index_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed {file_path}")

        # Clear current_index directory
        if os.path.exists(current_index_dir):
            for file in os.listdir(current_index_dir):
                file_path = os.path.join(current_index_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed {file_path}")

        # Create empty chunks.txt file
        chunks_file = os.path.join(current_index_dir, "chunks.txt")
        with open(chunks_file, "w") as f:
            f.write("# Chunks file for current index\n")
        logger.info(f"Created empty chunks.txt file at {chunks_file}")

        # Initialize document processor
        processor = DocumentProcessor()

        # Create a new empty index
        processor.create_empty_index()

        # Save the empty index
        processor.save_index(index_dir)
        logger.info(f"Created and saved new empty index at {index_dir}")

        return True
    except Exception as e:
        logger.error(f"Error rebuilding index: {str(e)}")
        return False


if __name__ == "__main__":
    import time

    logger.info("Starting index rebuild")
    success = rebuild_index()
    if success:
        logger.info("Successfully rebuilt index")
        sys.exit(0)
    else:
        logger.error("Failed to rebuild index")
        sys.exit(1)

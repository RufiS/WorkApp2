#!/usr/bin/env python3
# Script to fix the current_index directory issue

import os
import sys
import logging

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import resolve_path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fix_current_index():
    """
    Fix the current_index directory issue by creating it if it doesn't exist
    """
    try:
        # Resolve the path to the current_index directory
        current_index_dir = resolve_path("current_index", create_dir=True)

        # Create the directory if it doesn't exist
        if not os.path.exists(current_index_dir):
            os.makedirs(current_index_dir, exist_ok=True)
            logger.info(f"Created current_index directory at {current_index_dir}")
        else:
            logger.info(f"current_index directory already exists at {current_index_dir}")

        # Create an empty chunks.txt file if it doesn't exist
        chunks_file = os.path.join(current_index_dir, "chunks.txt")
        if not os.path.exists(chunks_file):
            with open(chunks_file, "w") as f:
                f.write("# Chunks file for current index\n")
            logger.info(f"Created empty chunks.txt file at {chunks_file}")
        else:
            logger.info(f"chunks.txt file already exists at {chunks_file}")

        return True
    except Exception as e:
        logger.error(f"Error fixing current_index directory: {str(e)}")
        return False


if __name__ == "__main__":
    logger.info("Starting current_index directory fix")
    success = fix_current_index()
    if success:
        logger.info("Successfully fixed current_index directory")
        sys.exit(0)
    else:
        logger.error("Failed to fix current_index directory")
        sys.exit(1)

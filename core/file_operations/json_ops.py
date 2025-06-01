"""Modern JSON Operations for WorkApp2.

Uses orjson for performance and proper error handling.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union
import json

logger = logging.getLogger(__name__)

# Try to use orjson for better performance, fall back to standard json
try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False
    logger.info("orjson not available, using standard json module")


class JsonHandler:
    """Modern JSON operations with performance optimization."""

    @staticmethod
    def loads(json_str: str) -> Optional[Any]:
        """Parse JSON string with proper error handling.

        Args:
            json_str: JSON string to parse

        Returns:
            Parsed object or None if error
        """
        try:
            if HAS_ORJSON:
                return orjson.loads(json_str)
            else:
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing JSON: {str(e)}")
            return None

    @staticmethod
    def dumps(obj: Any, pretty: bool = False) -> Optional[str]:
        """Serialize object to JSON string.

        Args:
            obj: Object to serialize
            pretty: Whether to pretty-print the JSON

        Returns:
            JSON string or None if error
        """
        try:
            if HAS_ORJSON:
                option = orjson.OPT_INDENT_2 if pretty else 0
                return orjson.dumps(obj, option=option).decode('utf-8')
            else:
                if pretty:
                    return json.dumps(obj, indent=2, ensure_ascii=False)
                else:
                    return json.dumps(obj, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            logger.error(f"Error serializing to JSON: {str(e)}")
            return None

    @staticmethod
    def load_from_file(file_path: Union[str, Path]) -> Optional[Any]:
        """Load JSON from file with proper error handling.

        Args:
            file_path: Path to JSON file

        Returns:
            Parsed object or None if error
        """
        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"JSON file does not exist: {path}")
                return None

            content = path.read_text(encoding='utf-8')
            return JsonHandler.loads(content)

        except Exception as e:
            logger.error(f"Error loading JSON from {file_path}: {str(e)}")
            return None

    @staticmethod
    def save_to_file(obj: Any, file_path: Union[str, Path], pretty: bool = True) -> bool:
        """Save object to JSON file.

        Args:
            obj: Object to save
            file_path: Path to save file
            pretty: Whether to pretty-print the JSON

        Returns:
            True if successful, False otherwise
        """
        try:
            from .path_utils import PathUtils

            path = PathUtils.resolve_path(file_path, create_dir=True)
            json_str = JsonHandler.dumps(obj, pretty=pretty)

            if json_str is None:
                return False

            path.write_text(json_str, encoding='utf-8')
            logger.debug(f"Successfully saved JSON to: {path}")
            return True

        except Exception as e:
            logger.error(f"Error saving JSON to {file_path}: {str(e)}")
            return False

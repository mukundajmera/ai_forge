"""Persistent Storage - JSON-based persistence for AI Forge data.

This module provides a simple JSON file-based storage mechanism to ensure
data sources and datasets persist across backend restarts.
"""

import json
import logging
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class PersistentStorage:
    """Thread-safe persistent storage using JSON file."""

    def __init__(self, storage_path: str = "./data/storage.json"):
        """Initialize persistent storage.

        Args:
            storage_path: Path to the JSON storage file.
        """
        self.storage_path = Path(storage_path)
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = {
            "data_sources": {},
            "parsed_files": {},
            "datasets": {},
            "jobs": {},
        }
        self._load()

    def _load(self) -> None:
        """Load data from storage file."""
        with self._lock:
            if not self.storage_path.exists():
                logger.info(f"No existing storage found at {self.storage_path}, starting fresh.")
                self._save()  # Initialize file
                return

            try:
                content = self.storage_path.read_text(encoding="utf-8")
                if not content.strip():
                     self._save()
                     return
                
                loaded_data = json.loads(content)
                self._data.update(loaded_data)
                
                # Ensure all required keys exist
                for key in ["data_sources", "parsed_files", "datasets", "jobs"]:
                    if key not in self._data:
                        self._data[key] = {}
                        
                logger.info(f"Loaded storage from {self.storage_path}")
            except Exception as e:
                logger.error(f"Failed to load storage from {self.storage_path}: {e}")
                # Backup corrupt file
                if self.storage_path.exists():
                    backup_path = self.storage_path.with_suffix(".json.bak")
                    shutil.copy(self.storage_path, backup_path)
                    logger.warning(f"Backed up corrupt storage to {backup_path}")

    def _save(self) -> None:
        """Save data to storage file."""
        with self._lock:
            try:
                self.storage_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Atomic write
                temp_path = self.storage_path.with_suffix(".tmp")
                temp_path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
                temp_path.replace(self.storage_path)
                
            except Exception as e:
                logger.error(f"Failed to save storage to {self.storage_path}: {e}")

    # -------------------------------------------------------------------------
    # Generic Accessors
    # -------------------------------------------------------------------------

    def get_all(self, collection: str) -> Dict[str, Any]:
        """Get all items from a collection."""
        with self._lock:
            return self._data.get(collection, {})

    def get(self, collection: str, item_id: str) -> Optional[Any]:
        """Get an item by ID."""
        with self._lock:
            return self._data.get(collection, {}).get(item_id)

    def set(self, collection: str, item_id: str, value: Any) -> None:
        """Set an item."""
        with self._lock:
            if collection not in self._data:
                self._data[collection] = {}
            self._data[collection][item_id] = value
            self._save()

    def delete(self, collection: str, item_id: str) -> None:
        """Delete an item."""
        with self._lock:
            if collection in self._data and item_id in self._data[collection]:
                del self._data[collection][item_id]
                self._save()
                
    def clear(self, collection: str) -> None:
        """Clear a collection."""
        with self._lock:
             self._data[collection] = {}
             self._save()

# Global instance
storage = PersistentStorage()

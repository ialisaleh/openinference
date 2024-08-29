import os

from arize.experimental.datasets import ArizeDatasetsClient


ARIZE_DEVELOPER_KEY = os.getenv("ARIZE_DEVELOPER_KEY", None)
ARIZE_SPACE_ID = os.getenv("ARIZE_SPACE_ID", None)
ARIZE_DATASET_NAME = os.getenv("ARIZE_DATASET_NAME", "jailbreak_prompts")


def get_arize_datasets_client():
    """
    Returns an instance of ArizeDatasetsClient using the provided developer key.
    """
    return ArizeDatasetsClient(developer_key=ARIZE_DEVELOPER_KEY)


class MemoryStore:
    def __init__(self):
        """Initializes the MemoryStore with an empty list."""
        self._data = []

    def add_item(self, item: str):
        """Adds a single item to the store."""
        self._data.append(item)

    def add_items(self, items: list):
        """Adds multiple items to the store."""
        self._data.extend(items)

    def remove_item(self, item: str):
        """Removes an item from the store if it exists."""
        if item in self._data:
            self._data.remove(item)

    def get_items(self) -> list:
        """Returns all items in the store."""
        return self._data

    def compare_with(self, other_list: list) -> bool:
        """Compares the store's data with another list."""
        return self._data == other_list

    def clear(self):
        """Clears all items from the store."""
        self._data.clear()

    def contains(self, item: str) -> bool:
        """Checks if an item is in the store."""
        return item in self._data

    def get_item_at(self, index: int) -> str:
        """Gets the item at a specific index, returns None if index is out of range."""
        return self._data[index] if 0 <= index < len(self._data) else None

    def length(self) -> int:
        """Returns the number of items in the store."""
        return len(self._data)

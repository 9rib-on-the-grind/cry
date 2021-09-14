"""Module for handling nested data structures."""

from __future__ import annotations
import collections
from collections.abc import Iterable, Sequence
import time



class DataMaintainer:
    """Data Maintainer class for efficient data stream update.

    Args:
        _data: Dictionary that maps key to another DataMaintainer or any other object.
    """

    def __init__(self):
        self._data = collections.defaultdict(DataMaintainer)
        self.update_hash = None
    
    def __getitem__(self, keys: Sequence) -> DataMaintainer:
        if not isinstance(keys, (tuple, list)):
            keys = (keys,)
        key, *other = keys
        if key not in self._data:
            raise KeyError(key)
        return self._data[key] if not other else self._data[key].__getitem__(other)

    def construct_location(self, keys: Sequence) -> DataMaintainer:
        key, *other = keys
        return self._data[key] if not other else self._data[key].construct_location(other)

    def add(self, mapping: dict, 
                  location: Sequence = None):
        subunit = self.construct_location(location) if location else self
        for key, val in mapping.items():
            subunit._data[key] = val

    def drop(self, key, recursively: bool = False):
        if not recursively or key in self._data:
            self._data.pop(key)
        else:
            for obj in self.values():
                if isinstance(obj, DataMaintainer):
                    obj.drop(key, recursively=True)

    def update(self, data: Iterable, keys: Iterable = 'auto'):
        """Update items by keys.

        Args:
            data: Iterable. New values.
            keys: Iterable (optional). Keys to be updated.
        """

        keys = self._data.keys() if keys == 'auto' else keys
        for key, val in zip(keys, data):
            self._data[key] = val
        self.set_update_hash()

    def set_update_hash(self, val=None):
        """Set hash representation of last update time."""
        self.update_hash = val if val is not None else time.time()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def show(self, location='root', indentatioin=0):
        maintainers, ordinary = [], []
        for key, obj in self._data.items():
            lst = maintainers if isinstance(obj, DataMaintainer) else ordinary
            lst.append((key, obj))
        print(' ' * indentatioin + location)
        for key, maintainer in maintainers:
            maintainer.show(f'{location}/{key}', indentatioin + 10)
        for key, val in ordinary:
            print(' ' * (indentatioin + 10) + f'{key} : {val}')
import collections




class DataMaintainer:
    """Data Maintainer class for efficient data stream update.

    Args:
        _data: Dictionary that maps key to another DataMaintainer instance or deque.
        maxlen: Int. Maximum length of deque for columns.
    """

    def __init__(self, maxlen=2000):
        self._data = collections.defaultdict(DataMaintainer)
        self.maxlen = maxlen
        self.update_hash = None
    
    def __getitem__(self, keys):
        if not isinstance(keys, (tuple, list)):
            keys = (keys,)
        key, *other = keys
        if key not in self._data:
            raise KeyError(key)
        return self._data[key] if not other else self._data[key].__getitem__(other)

    def construct_location(self, keys):
        key, *other = keys
        return self._data[key] if not other else self._data[key].construct_location(other)

    def add(self, data=None, keys=[], location=None, maxlen=None):
        """Add data into the DataMaintainer.

        Args:
            data: Iterable. Contains columns to add.
            keys: Iterable. Contains keys for columns in same order.
            location: List. Contains subsequent keys for multikey access.
            maxlen: Int. Maximum length of deque for columns.
        """

        subunit = self.construct_location(location) if location else self
        subunit.maxlen = maxlen if maxlen is not None else self.maxlen
        for key, column in zip(keys, data if data is not None else [[] * len(keys)]):
            subunit._data[key] = collections.deque(column, maxlen=subunit.maxlen)
        return subunit

    def append(self, data, keys='auto'):
        """Append elements in data to deque specified by key.

        Args:
            data: Iterable. Elements to be added.
            keys: Iterable (optional). Contains keys for columns in same order.
                  By default keys are in the same order they were added.
        """
        keys = self._data.keys() if keys == 'auto' else keys
        for key, value in zip(keys, data):
            self._data[key].append(value)

    def set_update_hash(self, val):
        """Set hash representation of last time when data was updated"""
        self.update_hash = val

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def iterrows(self):
        for row in zip(*self.values()):
            return row

    def show(self, location='', show_last=10):
        maintainers, deques = [], []
        for key, obj in self._data.items():
            lst = maintainers if isinstance(obj, DataMaintainer) else deques
            lst.append((key, obj))
        
        for key, maintainer in maintainers:
                maintainer.show(f'{location}/{key}')
        print('\n' + location)
        for key, _ in deques:
            print(f'{key:>19}', end='')
        print('\n' * 2)
        cols = [list(val)[-show_last:] for key, val in deques]
        for attrs in zip(*cols):
            for val in attrs:
                print('{:>19.5f}'.format(val), end='')
            print()
class ListNode:
    """
    Nodes are defined for the doubly-linked list.
    """
    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.next = self.prev = None

    def __repr__(self):
        return f'{self.key}, {self.val}'

    def __str__(self):
        return self.__repr__()


class LRUCache:
    """
    Least Recently Used Cache stores key-value pairs up to a capacity, after which it removes the least recently
    accessed entries to accommodate additional entries.
    This implementation utilizes a doubly-linked list and dictionary.
    The doubly-linked list orders the data formatted as nodes, allowing add and remove in constant O(1) time.
    The dictionary stores nodes as values, enabling constant O(1) time access to a node when given the key.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.head = ListNode()  # Node preceding cache
        self.tail = ListNode()  # Node following cache
        self.head.next, self.tail.prev = self.tail, self.head  # Double linkage
        self.cache_dict = {}

    def __repr__(self):
        repr = []
        current = self.head.next  # Skip head
        while current.next:  # Finish before tail
            repr.extend([f'{current.key}, {current.val}', ' <-> '])
            current = current.next
        return ''.join(repr[:-1])

    def __str__(self):
        return self.__repr__()

    def get(self, key):
        """
        Returns the value corresponding to the key input in constant O(1) time
        """
        if key not in self.cache_dict:
            print("Key not found.")
            return None
        node = self.cache_dict[key]
        self._remove(node)
        self._add(node)
        return node.val

    def put(self, key, value):
        """
        Adds new node to cache while removing oldest node if capacity is exceeded, in constant O(1) time
        """
        node = ListNode(key, value)
        if key in self.cache_dict:
            self._remove(self.cache_dict[key])
        self._add(node)
        self.cache_dict[key] = node
        if len(self.cache_dict) > self.capacity:
            del self.cache_dict[self.head.next.key]
            self._remove(self.head.next)

    def _add(self, node):
        """
        Inserts node between end of cache and tail node, maintaining double linkages
        """
        self.tail.prev.next = node
        node.prev = self.tail.prev
        node.next = self.tail
        self.tail.prev = node

    def _remove(self, node):
        """
        Removes node, maintaining double linkages
        """
        node.next.prev = node.prev
        node.prev.next = node.next

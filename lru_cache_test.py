from lru_cache import ListNode, LRUCache
import unittest


class LruCacheTest(unittest.TestCase):

    def setUp(self) -> None:
        self.cache = LRUCache(2)

    def test_empty_lru(self):
        self.assertIsNone(self.cache.get(1))
        self.assertIsNone(self.cache.get(2))
        self.assertIsNone(self.cache.get(-1))

    def test_add_one_key(self):
        key, value = 1, 2
        self.cache.put(key, value)
        self.assertEqual(self.cache.get(key), value)

        diff_key = 2
        self.assertIsNone(self.cache.get(diff_key))

    def test_capacity(self):
        self.cache.put(1, 2)
        self.cache.put(3, 4)
        self.cache.put(5, 6)
        self.assertIsNone(self.cache.get(1))
        self.assertIsNotNone(self.cache.get(3))
        self.assertIsNotNone(self.cache.get(5))
        self.assertEqual(self.cache.capacity, 2)

    def test_complex(self):
        cache = LRUCache(5)
        cache.put(1, 2)
        cache.put(3, 4)
        self.assertIsNotNone(cache.get(1))
        cache.put(4, 5)
        cache.put(5, 6)
        cache.put(4, 6)
        cache.put(6, 7)
        cache.put(2, 3)
        self.assertIsNone(cache.get(3))
        self.assertIsNotNone(cache.get(2))
        self.assertEqual(cache.get(4), 6)
        self.assertEqual(cache.capacity, 5)

    def test_repr(self):
        node = ListNode(1, 2)
        self.assertEqual(repr(node), '1, 2')
        self.assertEqual(str(node), '1, 2')
        cache = LRUCache(3)
        cache.put(1, 2)
        cache.put(5, 6)
        cache.put(3, 4)
        self.assertEqual(repr(cache), '1, 2 <-> 5, 6 <-> 3, 4')
        self.assertEqual(str(cache), '1, 2 <-> 5, 6 <-> 3, 4')


if __name__ == "__main__":
    unittest.main()

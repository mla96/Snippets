from binary_tree_serializer import TreeNode, BinaryTreeSerializer
import unittest
from copy import deepcopy


class BinaryTreeSerializerTest(unittest.TestCase):

    def setUp(self) -> None:
        """
        Sample binary tree and equivalent string representation
        """
        self.bts = BinaryTreeSerializer()

        self.node = TreeNode(1)
        self.node.left, self.node.right = TreeNode(2), TreeNode(3)
        self.node.right.left, self.node.right.right = TreeNode(4), TreeNode(5)
        self.str_repr = '1,2,3,x,x,4,5,x,x,x,x'

    def test_serialize(self):
        self.assertEqual(self.bts.serialize(self.node), self.str_repr)

    def test_deserialize(self):
        self.assertTrue(self.is_same_binary_tree(self.bts.deserialize(self.str_repr), self.node))

    def test_recover_binary_tree(self):
        """
        Tests serialization and deserialization sequentially
        """
        self.assertTrue(self.is_same_binary_tree(self.bts.deserialize(self.bts.serialize(self.node)), self.node))
        self.assertEqual(self.bts.serialize(self.bts.deserialize(self.str_repr)), self.str_repr)

    def is_same_binary_tree(self, node1, node2):
        """
        Tests whether two binary trees are identical; necessary because they occupy different memory addresses
        Operates in linear O(n) time via recursion
        """
        if not node1 and not node2:
            return True

        if node1 and node2:
            is_same_curr_node = (node1.val == node2.val)
            is_same_left_node = self.is_same_binary_tree(node1.left, node2.left)
            is_same_right_node = self.is_same_binary_tree(node1.right, node2.right)
            return is_same_curr_node and is_same_left_node and is_same_right_node
        return False

    def test_is_same_binary_tree(self):
        """
        Tests is_same_binary_tree helper function.
        """
        self.assertTrue(self.is_same_binary_tree(self.node, deepcopy(self.node)))

        left_only = TreeNode(1)
        left_only.left = TreeNode(2)
        right_only = TreeNode(1)
        right_only.right = TreeNode(2)
        self.assertFalse(self.is_same_binary_tree(left_only, right_only))

        self.assertFalse(self.is_same_binary_tree(TreeNode(0), TreeNode(1)))
        self.assertTrue(self.is_same_binary_tree(TreeNode(1), TreeNode(1)))
        self.assertFalse(self.is_same_binary_tree(TreeNode(1), right_only))


if __name__ == "__main__":
    unittest.main()

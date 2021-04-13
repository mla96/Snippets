import collections


class TreeNode:
    """
    Nodes are defined for the binary tree.
    """
    def __init__(self, x):
        self.val = x
        self.left = self.right = None


class BinaryTreeSerializer:
    """
    Converts a binary tree to a string, then reconstructs the original binary tree using that string.
    This is done in O(n) time because it requires a single traversal across all nodes for either representation.
    Example of tree: [1,2,3,null,null,4,5]
        1 is root and has children: 2, 3
        2 has no children: null, null
        3 has children: 4, 5
    Example of string: "1,2,3,x,x,4,5,x,x,x,x"
    """
    def serialize(self, root):
        """
        Encodes a tree to a single string using iterative level traversal.
        """
        data = ""
        queue = collections.deque([root])
        while queue:
            node = queue.popleft()
            if node:
                data += str(node.val) + ","
                queue.append(node.left)
                queue.append(node.right)
            else:
                data += "x" + ","
        return data[:-1]

    def deserialize(self, data):
        """
        Decodes encoded data to tree using iterative level traversal.
        """
        if not data or data == "x":
            return []
        data = data.split(",")
        root = TreeNode(int(data[0]))
        queue = collections.deque([root])
        data = data[1:]
        while data:
            node = queue.popleft()
            if data[0] != "x":
                node.left = TreeNode(int(data[0]))
                queue.append(node.left)
            if data[1] != "x":
                node.right = TreeNode(int(data[1]))
                queue.append(node.right)
            data = data[2:]
        return root

from typing import Optional
from bitarray import bitarray

class AdaptiveHuffmanNode:
    _order_counter = 512  # Start high and count down
    
    def __init__(self, parent: Optional['AdaptiveHuffmanNode'] = None, left: Optional['AdaptiveHuffmanNode'] = None, right: Optional['AdaptiveHuffmanNode'] = None, weight: int = 0, char: Optional[str] = None):
        self.parent = parent
        self.left = left
        self.right = right
        self.weight = weight
        self.char = char
        self.order = AdaptiveHuffmanNode._order_counter
        AdaptiveHuffmanNode._order_counter -= 1

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def __repr__(self):
        return f"Node('{self.char}', w={self.weight}, o={self.order})"

class AdaptiveHuffmanCompressor:
    def __init__(self):
        AdaptiveHuffmanNode._order_counter = 512
        self.NYT = AdaptiveHuffmanNode(char='NYT', weight=0)
        self.root = self.NYT
        self.char_to_node = {}
        self.all_nodes = [self.NYT]

    def _get_path(self, node):
        """Get the binary path from root to a node."""
        path = []
        current = node
        while current.parent is not None:
            if current.parent.left == current:
                path.append(0)
            else:
                path.append(1)
            current = current.parent
        path.reverse()
        return bitarray(path)

    def _find_leader(self, node):
        """Find the highest order node with the same weight."""
        leader = node
        for n in self.all_nodes:
            if n.weight == node.weight and n.order > leader.order:
                leader = n
        return leader
    
    def _swap_nodes(self, node1, node2):
        """Swap two nodes in the tree."""
        if node1 == node2:
            return
        
        # Swap parents
        p1, p2 = node1.parent, node2.parent
        
        if p1 == p2:
            # Same parent - just swap left/right
            p1.left, p1.right = p1.right, p1.left
        else:
            # Different parents
            if p1:
                if p1.left == node1:
                    p1.left = node2
                else:
                    p1.right = node2
            else:
                self.root = node2
                
            if p2:
                if p2.left == node2:
                    p2.left = node1
                else:
                    p2.right = node1
            else:
                self.root = node1
            
            node1.parent, node2.parent = p2, p1
        
        # Swap orders
        node1.order, node2.order = node2.order, node1.order

    def _update_tree(self, leaf):
        """Update tree using Vitter's FGK algorithm with improved efficiency."""
        current = leaf
        while current is not None:
            # Find the leader of current's weight block
            leader = self._find_leader(current)
            
            # Swap with leader if different and not parent
            if current != leader and leader != current.parent:
                self._swap_nodes(current, leader)
            
            # Increment weight
            current.weight += 1
            current = current.parent

    def compress_stream(self, text):
        """Compresses a string of text and returns the bitstring."""
        if not text:
            print("Warning: Empty input!")
            return '', 0

        print(f"Compressing {len(text)} characters...")
        compressed = bitarray()

        for char in text:
            if char in self.char_to_node:
                # Character seen before - send its code and update
                node = self.char_to_node[char]
                path = self._get_path(node)
                compressed.extend(path)
                self._update_tree(node)
            else:
                # First occurrence - send NYT code + character
                nyt_path = self._get_path(self.NYT)
                compressed.extend(nyt_path)
                
                # Send 8-bit character code
                char_code = ord(char)
                for i in range(7, -1, -1):
                    compressed.append((char_code >> i) & 1)
                
                # Split NYT into two children
                old_nyt = self.NYT
                
                # New character node (right child) with weight 1
                new_char_node = AdaptiveHuffmanNode(parent=old_nyt, char=char, weight=1)
                # New NYT (left child) with weight 0
                new_nyt = AdaptiveHuffmanNode(parent=old_nyt, char='NYT', weight=0)
                
                # Update old NYT to become internal node
                old_nyt.left = new_nyt
                old_nyt.right = new_char_node
                old_nyt.char = None
                old_nyt.weight = 1  # Internal node weight is sum of children
                
                # Update tracking
                self.NYT = new_nyt
                self.char_to_node[char] = new_char_node
                self.all_nodes.extend([new_char_node, new_nyt])
                
                # Update tree from old_nyt upwards
                self._update_tree(old_nyt)

        total_bits = len(compressed)
        bit_string = compressed.to01()
        print(f"Total compressed size: {total_bits} bits ({total_bits//8} bytes)")
        return bit_string, total_bits
    


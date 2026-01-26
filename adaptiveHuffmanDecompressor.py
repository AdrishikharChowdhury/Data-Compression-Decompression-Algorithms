# adaptiveHuffmanDecompressor.py - Adaptive Huffman decompression functionality
from typing import Optional
from bitarray import bitarray
from file_handler import write_text_file
import os

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

class AdaptiveHuffmanDecompressor:
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

    def decompress_from_bits(self, bit_string, original_length):
        """Decompress from bit string and return original text."""
        if not bit_string:
            return ""
        
        # Reset tree for decompression - start fresh
        self.__init__()
        
        bits = bitarray(bit_string)
        result = []
        i = 0
        
        while i < len(bits) and len(result) < original_length:
            # Start from root and follow path
            current = self.root
            
            # If we're at NYT initially, this is a new character
            if current == self.NYT:
                # Read 8-bit character directly
                if i + 8 > len(bits):
                    break
                char_code = 0
                for j in range(8):
                    char_code = (char_code << 1) | bits[i + j]
                i += 8
                char = chr(char_code)
                result.append(char)
                
                # Update tree with this new character
                old_nyt = self.NYT
                new_char_node = AdaptiveHuffmanNode(parent=old_nyt, char=char, weight=1)
                new_nyt = AdaptiveHuffmanNode(parent=old_nyt, char='NYT', weight=0)
                
                old_nyt.left = new_nyt
                old_nyt.right = new_char_node
                old_nyt.char = None
                old_nyt.weight = 1
                
                self.NYT = new_nyt
                self.char_to_node[char] = new_char_node
                self.all_nodes.extend([new_char_node, new_nyt])
                
                self._update_tree(old_nyt)
                continue
            
            # Navigate through tree
            while current is not None and current != self.NYT and not current.is_leaf() and i < len(bits):
                if bits[i] == 0:
                    current = current.left
                else:
                    current = current.right
                i += 1
            
            if current == self.NYT:
                # Read 8-bit character
                if i + 8 > len(bits):
                    break
                char_code = 0
                for j in range(8):
                    char_code = (char_code << 1) | bits[i + j]
                i += 8
                char = chr(char_code)
                result.append(char)
                
                # Update tree with this new character
                old_nyt = self.NYT
                new_char_node = AdaptiveHuffmanNode(parent=old_nyt, char=char, weight=1)
                new_nyt = AdaptiveHuffmanNode(parent=old_nyt, char='NYT', weight=0)
                
                old_nyt.left = new_nyt
                old_nyt.right = new_char_node
                old_nyt.char = None
                old_nyt.weight = 1
                
                self.NYT = new_nyt
                self.char_to_node[char] = new_char_node
                self.all_nodes.extend([new_char_node, new_nyt])
                
                self._update_tree(old_nyt)
            elif current is not None and current.is_leaf():
                result.append(current.char)
                self._update_tree(current)
        
        return ''.join(result)
    
    def decompress_from_file(self, input_path):
        """Decompress from file and return original text."""
        with open(input_path, 'rb') as f:
            # Read original length
            original_length = int.from_bytes(f.read(8), 'big')
            
            # Read compressed bits
            bits = bitarray()
            bits.fromfile(f)
            
            return self.decompress_from_bits(bits.to01(), original_length)

# --- File paths and public functions ---
filePath = "./files"
outputFiles = f"{filePath}/outputs"
outputAdaptiveHuffmannFiles = f"{outputFiles}/adaptive_huffman_files"

def adaptiveHuffmanDecompression():
    """Decompress an Adaptive Huffman compressed file."""
    input_file = f"{outputAdaptiveHuffmannFiles}/adaptive_test.ahuf"
    output_file = f"{outputAdaptiveHuffmannFiles}/adaptive_decompressed.txt"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    
    print("Decompressing Adaptive Huffman file...")
    decompressor = AdaptiveHuffmanDecompressor()
    decompressed_text = decompressor.decompress_from_file(input_file)
    
    write_text_file(output_file, decompressed_text)
    print(f"Adaptive Huffman decompression complete. Output saved to: {output_file}")
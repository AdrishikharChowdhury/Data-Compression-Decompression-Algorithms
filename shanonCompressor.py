# shannon_fano.py - Shannon-Fano algorithm (top-down splitting)
from collections import Counter

class ShannonFanoCompressor:
    def build_codes(self, text):
        """Shannon-Fano: Sort → Split → Assign 0/1 recursively"""
        freq = Counter(text)
        symbols = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        codes = {}
        self._shannon_split(symbols, "", codes)
        return codes
    
    def _shannon_split(self, symbols, prefix, codes):
        """Recursively split into equal probability groups"""
        if len(symbols) == 1:
            codes[symbols[0][0]] = prefix
            return
        
        # Find best split point (equal probability groups)
        total_prob = sum(freq for _, freq in symbols)
        left_prob = 0
        
        for i in range(1, len(symbols)):
            left_prob += symbols[i-1][1]
            right_prob = total_prob - left_prob
            
            if abs(left_prob - right_prob) < symbols[0][1]:  # Good split
                break
        
        # Split: left=1, right=0 (or vice versa)
        left_group = symbols[:i]
        right_group = symbols[i:]
        
        self._shannon_split(left_group, prefix + "1", codes)
        self._shannon_split(right_group, prefix + "0", codes)
    
    def compress(self, text):
        """Compress text using Shannon-Fano codes"""
        codes = self.build_codes(text)
        compressed = ''.join(codes[char] for char in text)
        return compressed, codes

# For tree reconstruction (decompression) - simplified
def build_shannon_tree(codes):
    """Build tree from codes for decompression"""
    class Node:
        def __init__(self, char=None):
            self.char = char
            self.left = self.right = None
    
    root = Node()
    for char, code in codes.items():
        node = root
        for bit in code:
            if bit == '0':
                if not node.left: node.left = Node()
                node = node.left
            else:
                if not node.right: node.right = Node()
                node = node.right
        node.char = char
    return root

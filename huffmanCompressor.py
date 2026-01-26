# compressor.py - Pure Huffman functions
from collections import Counter
import heapq

from typing import Optional

class Node:
    def __init__(self, char: Optional[str] = None, freq: int = 0):
        self.char = char
        self.freq = freq
        self.left: Optional['Node'] = None
        self.right: Optional['Node'] = None
    
    def __lt__(self, other):
        return self.freq < other.freq

class HuffmanCompressor:
    def build_tree(self, freq):
        heap = [Node(char, f) for char, f in freq.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = Node(None, left.freq + right.freq)
            merged.left, merged.right = left, right
            heapq.heappush(heap, merged)
        return heap[0]
    
    def generate_codes(self, node, code='', codes=None):
        if codes is None:
            codes = {}
        if node.char is not None:
            codes[node.char] = code
            return codes
        if node.left:
            self.generate_codes(node.left, code + '0', codes)
        if node.right:
            self.generate_codes(node.right, code + '1', codes)
        return codes
    
    def compress(self, text):
        if not text:
            return "", None, {}
        freq = Counter(text)
        root = self.build_tree(freq)
        codes = self.generate_codes(root, '', {})
        compressed = ''.join(codes[char] for char in text)
        return compressed, root, codes


    
    def compress_file(self, text, output_path):
        if not text:
            with open(output_path, 'wb') as f:
                f.write(b'')
            return

        freq = Counter(text)
        root = self.build_tree(freq)
        codes = self.generate_codes(root, '', {})
        compressed_bits_str = ''.join(codes[char] for char in text)

        # Use efficient bit packing with bitarray
        from bitarray import bitarray
        bits = bitarray(compressed_bits_str)
        
        # Pad to full byte with minimal padding
        padding = (8 - len(bits) % 8) % 8
        if padding > 0:
            bits.extend([0] * padding)

        with open(output_path, 'wb') as f:
            # Write frequency table size
            f.write(len(freq).to_bytes(4, 'big'))
            
            # Write frequency table more efficiently
            for char, f_val in freq.items():
                char_bytes = char.encode('utf-8')
                f.write(len(char_bytes).to_bytes(1, 'big'))  # Char length
                f.write(char_bytes)
                f.write(f_val.to_bytes(4, 'big'))
            
            # Write padding info
            f.write(padding.to_bytes(1, 'big'))
            
            # Write compressed data
            bits.tofile(f)



# huffmanDecompressor.py - Huffman decompression functionality
from collections import Counter
import heapq
from typing import Optional, Dict
from file_handler import write_text_file
import os

class Node:
    def __init__(self, char: Optional[str] = None, freq: int = 0):
        self.char = char
        self.freq = freq
        self.left: Optional['Node'] = None
        self.right: Optional['Node'] = None
    
    def __lt__(self, other):
        return self.freq < other.freq

class HuffmanDecompressor:
    def build_tree(self, freq):
        """Build Huffman tree from frequency table."""
        heap = [Node(char, f) for char, f in freq.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = Node(None, left.freq + right.freq)
            merged.left, merged.right = left, right
            heapq.heappush(heap, merged)
        return heap[0]
    
    def decompress(self, compressed_bits, root):
        """Decompress using the Huffman tree."""
        if not compressed_bits:
            return ''
        
        result = []
        current = root
        
        for bit in compressed_bits:
            # Traverse tree based on bit
            if bit == '0':
                current = current.left
            else:
                current = current.right
            
            # If we reach a leaf, we found a character
            if current.char is not None:
                result.append(current.char)
                current = root  # Reset to root for next character
        
        return ''.join(result)
    
    def decompress_from_file(self, input_path):
        """Decompress from file and return original text."""
        with open(input_path, 'rb') as f:
            # Read frequency table size
            freq_size = int.from_bytes(f.read(4), 'big')
            
            # Read frequency table
            freq = {}
            for _ in range(freq_size):
                char_len = int.from_bytes(f.read(1), 'big')
                char_bytes = f.read(char_len)
                char = char_bytes.decode('utf-8')
                freq_val = int.from_bytes(f.read(4), 'big')
                freq[char] = freq_val
            
            # Read padding info
            padding = int.from_bytes(f.read(1), 'big')
            
            # Read compressed data
            from bitarray import bitarray
            bits = bitarray()
            bits.fromfile(f)
            
            # Remove padding
            if padding > 0:
                bits = bits[:-padding]
            
            # Rebuild tree and decompress
            root = self.build_tree(freq)
            compressed_str = bits.to01()
            return self.decompress(compressed_str, root)

# --- File paths and public functions ---
filePath = "./files"
outputFiles = f"{filePath}/outputs"
outputHuffmanFiles = f"{outputFiles}/huffmann_files"

def huffmanDecompression():
    """Decompress a Huffman compressed file."""
    input_file = f"{outputHuffmanFiles}/huffman_test.huf"
    output_file = f"{outputHuffmanFiles}/huffman_decompressed.txt"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    
    print("Decompressing Huffman file...")
    decompressor = HuffmanDecompressor()
    decompressed_text = decompressor.decompress_from_file(input_file)
    
    write_text_file(output_file, decompressed_text)
    print(f"Huffman decompression complete. Output saved to: {output_file}")
# compressor.py - Pure Huffman functions
from collections import Counter
import heapq

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = self.right = None
    
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
    
    def generate_codes(self, node, code='', codes={}):
        if node.char is not None:
            codes[node.char] = code
            return codes
        self.generate_codes(node.left, code + '0', codes)
        self.generate_codes(node.right, code + '1', codes)
        return codes
    
    def compress(self, text):
        freq = Counter(text)
        root = self.build_tree(freq)
        codes = self.generate_codes(root)
        compressed = ''.join(codes[char] for char in text)
        return compressed, root, codes

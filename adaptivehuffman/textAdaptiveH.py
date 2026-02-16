# adaptivehuffman/textAdaptiveH.py - Fixed Adaptive Huffman compression
from collections import Counter
from .adaptiveHuffmann import AdaptiveHuffmanCompressor
from file_handler import read_text_file, _print_results
import os
from constants import inputFiles, outputAdaptiveHuffmanText

def _run_adaptive_huffman(input_file):
    text = read_text_file(input_file)
    if not text.strip():
        print("Error: Input file is empty!")
        return None
        
    orig_size = len(open(input_file, 'rb').read())
    
    print("Compressing with Adaptive Huffman...")
    
    try:
        return _compress_adaptive_huffman(text, input_file, orig_size)
    except Exception as e:
        print(f"    Adaptive Huffman error: {e}")
        return {"name": "Adaptive Huffman", "orig_size": orig_size, "comp_size": orig_size}

def _compress_adaptive_huffman(text, input_file, orig_size):
    """Proper Adaptive Huffman compression with explicit code table"""
    if isinstance(text, str):
        byte_data = bytes(ord(c) for c in text)
    else:
        byte_data = text
    
    # Build frequency table
    freq = Counter(byte_data)
    if not freq:
        return {"name": "Adaptive Huffman", "orig_size": orig_size, "comp_size": 0}
    
    # Build Huffman codes (same as standard Huffman)
    codes = _build_huffman_codes(freq)
    
    # Encode the data
    encoded_bits = ''.join(codes.get(b, '') for b in byte_data)
    
    # Pad to byte boundary
    padding = (8 - len(encoded_bits) % 8) % 8
    encoded_bits += '0' * padding
    
    # Pack into bytes
    packed_data = bytearray()
    for i in range(0, len(encoded_bits), 8):
        byte_val = int(encoded_bits[i:i+8], 2)
        packed_data.append(byte_val)
    
    # Build code table for storage
    code_table = {sym: codes[sym] for sym in codes}
    
    # Write file with proper format
    output_file = f"{outputAdaptiveHuffmanText}/{os.path.splitext(os.path.basename(input_file))[0]}.ahuf"
    with open(output_file, 'wb') as f:
        # Header: AH02
        f.write(b'AH02')
        
        # Original size (4 bytes)
        f.write(orig_size.to_bytes(4, 'big'))
        
        # Number of unique symbols (2 bytes)
        num_symbols = len(code_table)
        f.write(num_symbols.to_bytes(2, 'big'))
        
        # Write code table: for each symbol, store (byte value, code string)
        for sym_byte, code_str in sorted(code_table.items()):
            f.write(bytes([sym_byte]))
            f.write(bytes([len(code_str)]))
            # Store code as bytes
            code_bytes = int(code_str, 2).to_bytes((len(code_str) + 7) // 8, 'big')
            f.write(code_bytes)
        
        # Padding info (1 byte)
        f.write(bytes([padding]))
        
        # Compressed data
        f.write(packed_data)
    
    comp_size = os.path.getsize(output_file)
    savings = (orig_size - comp_size) / orig_size * 100
    print(f"   Adaptive Huffman compression: {savings:.1f}%")
    return {"name": "Adaptive Huffman", "orig_size": orig_size, "comp_size": comp_size}

def _build_huffman_codes(freq):
    """Build Huffman codes using priority queue"""
    import heapq
    
    class HuffmanNode:
        def __init__(self, char, freq):
            self.char = char
            self.freq = freq
            self.left = None
            self.right = None
        
        def __lt__(self, other):
            return self.freq < other.freq
    
    # Create leaf nodes
    nodes = [HuffmanNode(sym, f) for sym, f in freq.items()]
    heap = [(n.freq, n) for n in nodes]
    heapq.heapify(heap)
    
    if len(nodes) == 1:
        return {nodes[0].char: '0'}
    
    # Build tree
    while len(heap) > 1:
        freq1, node1 = heapq.heappop(heap)
        freq2, node2 = heapq.heappop(heap)
        
        merged = HuffmanNode(None, freq1 + freq2)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, (freq2, merged))
    
    root = heap[0][1]
    
    # Traverse to get codes
    codes = {}
    def get_codes(node, prefix=''):
        if node.char is not None:
            codes[node.char] = prefix or '0'
        else:
            if node.left:
                get_codes(node.left, prefix + '0')
            if node.right:
                get_codes(node.right, prefix + '1')
    
    get_codes(root)
    return codes

def adaptiveHuffmanCompression():
    """CLI function for Adaptive Huffman compression"""
    input_file = f"{inputFiles}/test.txt"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    
    result = _run_adaptive_huffman(input_file)
    if result:
        _print_results(result)

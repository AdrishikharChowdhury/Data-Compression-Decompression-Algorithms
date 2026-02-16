"""
Text Huffman Compression Module
Fixed Huffman compression for text files
"""

from file_handler import read_text_file, _print_results, read_binary_data
import os
from constants import inputFiles, outputHuffmanText
from huffman.huffmanCompressor import HuffmanCompressor
import heapq

def _run_huffman(input_file):
    """Runs Huffman compression."""
    text = read_text_file(input_file)
    orig_len = len(open(input_file, 'rb').read())
    
    print(f"Analyzing text ({orig_len} bytes) for Huffman compression...")
    
    try:
        # Use the HuffmanCompressor which uses proper HU02 format
        compressor = HuffmanCompressor()
        result = compressor.compress_file(text, f"{outputHuffmanText}/{os.path.splitext(os.path.basename(input_file))[0]}.huf")
        result["name"] = "Huffman"
        return result
    except Exception as e:
        print(f"    Huffman error: {e}")
        return {"name": "Huffman", "orig_size": orig_len, "comp_size": orig_len}

def huffmanCompression():
    """CLI function"""
    input_file = f"{inputFiles}/test.txt"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    
    result = _run_huffman(input_file)
    if result:
        _print_results(result)

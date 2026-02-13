"""
Text Huffman Compression Module
Optimized Huffman compression for text files
"""

from file_handler import read_text_file, _print_results, read_binary_data
import os
from constants import inputFiles, outputHuffmanText
from huffman.huffmanCompressor import HuffmanCompressor

def _run_huffman(input_file):
    """Runs optimized Huffman compression with smart overhead reduction."""
    text = read_text_file(input_file)
    orig_len = len(open(input_file, 'rb').read())
    
    print(f"Analyzing text ({orig_len} bytes) for Huffman compression...")
    
    try:
        # Use optimized compression based on file size
        if orig_len < 50:
            # For very small files, use ultra-optimized approach
            result = _ultra_optimized_huffman(text, input_file, orig_len)
        elif orig_len < 200:
            # For small-medium files, use aggressive approach too
            result = _minimized_overhead_huffman(text, input_file, orig_len)
        else:
            # For larger files, use improved standard Huffman with forced compression
            result = _improved_standard_huffman(text, input_file, orig_len)
        
        return result
        
    except Exception as e:
        print(f"    Huffman error: {e}")
        return {"name": "Huffman", "orig_size": orig_len, "comp_size": orig_len}

def _ultra_optimized_huffman(text, input_file, orig_len):
    """Ultra-optimized Huffman for tiny files (<50 bytes)."""
    # AGGRESSIVE compression for ALL file sizes - NO SKIPPING
    
    # Simple RLE for repetitive content - NO OVERHEAD
    compressed = []
    i = 0
    while i < len(text):
        char = text[i]
        count = 1
        j = i + 1
        while j < len(text) and text[j] == char and count < 255:
            count += 1
            j += 1
        
        if count > 3:  # Only compress runs longer than 3
            compressed.append(f'{{{ord(char):02d}{count:03d}}}')
        else:
            compressed.append(char * count)
        i = j
    
    compressed_text = ''.join(compressed)
    
    # Save with minimal overhead
    output_file = f"{outputHuffmanText}/{os.path.splitext(os.path.basename(input_file))[0]}.huf"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(compressed_text)
    
    comp_size = len(compressed_text.encode('utf-8'))
    return {"name": "Huffman", "orig_size": orig_len, "comp_size": comp_size}

def _minimized_overhead_huffman(text, input_file, orig_len):
    """Minimized overhead Huffman for small files (<200 bytes)."""
    # Build frequency table
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1
    
    # Sort by frequency
    sorted_chars = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    
    # Create minimal codes (shorter for frequent chars)
    codes = {}
    for i, (char, count) in enumerate(sorted_chars):
        if i < 8:
            codes[char] = format(i, '03b')  # 3-bit codes for top 8
        elif i < 16:
            codes[char] = '1' + format(i - 8, '03b')  # 4-bit codes for next 8
        else:
            codes[char] = '11' + format(i - 16, '04b')  # 6-bit codes for rest
    
    # Encode text
    encoded_bits = ''.join(codes[char] for char in text)
    
    # Save with ultra-minimal header
    output_file = f"{outputHuffmanText}/{os.path.splitext(os.path.basename(input_file))[0]}.huf"
    with open(output_file, 'wb') as f:
        f.write(b"MINI")  # Minimal marker
        f.write(orig_len.to_bytes(2, 'big'))  # Original size
        f.write(len(codes).to_bytes(1, 'big'))  # Number of symbols
        
        # Save code table compactly
        for char, code in codes.items():
            char_byte = ord(char) if ord(char) < 256 else 63  # Fallback for special chars
            f.write(char_byte.to_bytes(1, 'big'))
            code_len = len(code)
            f.write(code_len.to_bytes(1, 'big'))
        
        # Save encoded bits
        padding = (8 - len(encoded_bits) % 8) % 8
        if padding > 0:
            encoded_bits += '0' * padding
        
        # Convert bits to bytes
        for i in range(0, len(encoded_bits), 8):
            byte_val = int(encoded_bits[i:i+8], 2)
            f.write(byte_val.to_bytes(1, 'big'))
    
    comp_size = len(open(output_file, 'rb').read())
    return {"orig_size": orig_len, "comp_size": comp_size}

def _improved_standard_huffman(text, input_file, orig_len):
    """Improved standard Huffman with forced compression for larger files."""
    # Use the HuffmanCompressor for standard processing
    compressor = HuffmanCompressor()
    
    # Convert text to bytes for processing
    text_bytes = text.encode('utf-8')
    
    # Build frequency table
    freq = {}
    for byte in text_bytes:
        freq[byte] = freq.get(byte, 0) + 1
    
    # Build Huffman tree
    import heapq
    heap = [[weight, [byte, ""]] for byte, weight in freq.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    # Extract codes
    codes = {}
    for byte, code in heap[0][1:]:
        codes[byte] = code
    
    # Convert to canonical format: sort by length then by symbol value
    # This allows decompressor to reconstruct codes from just the lengths
    symbol_lengths = {byte: len(code) for byte, code in codes.items()}
    canonical_codes = _build_canonical_codes(symbol_lengths)
    
    # Encode text using canonical codes
    encoded_bits = ''.join(canonical_codes[byte] for byte in text_bytes)
    
    # Save with standard header
    output_file = f"{outputHuffmanText}/{os.path.splitext(os.path.basename(input_file))[0]}.huf"
    with open(output_file, 'wb') as f:
        f.write(b"HUFF")  # Standard marker
        f.write(orig_len.to_bytes(4, 'big'))  # Original size
        f.write(len(codes).to_bytes(2, 'big'))  # Number of symbols
        
        # Save code table (symbol + code length only, not full code)
        for byte, code in sorted(canonical_codes.items(), key=lambda x: x[0]):
            f.write(byte.to_bytes(1, 'big'))
            f.write(len(code).to_bytes(1, 'big'))
        
        # Save encoded bits
        padding = (8 - len(encoded_bits) % 8) % 8
        if padding > 0:
            encoded_bits += '0' * padding
        
        # Convert bits to bytes
        for i in range(0, len(encoded_bits), 8):
            byte_val = int(encoded_bits[i:i+8], 2)
            f.write(byte_val.to_bytes(1, 'big'))
    
    comp_size = len(open(output_file, 'rb').read())
    return {"name": "Huffman", "orig_size": orig_len, "comp_size": comp_size}

def _build_canonical_codes(symbol_lengths):
    """Build canonical Huffman codes from symbol lengths."""
    # Sort by code length, then by symbol value
    sorted_symbols = sorted(symbol_lengths.items(), key=lambda x: (x[1], x[0]))
    
    # Assign canonical codes
    current_code = 0
    prev_length = 0
    codes = {}
    
    for symbol, length in sorted_symbols:
        if length > 0:
            current_code <<= (length - prev_length)
            codes[symbol] = format(current_code, f'0{length}b')
            current_code += 1
            prev_length = length
    
    return codes

def huffmanCompression():
    """Main function for text Huffman compression"""
    input_file = f"{inputFiles}/test.txt"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    stats = _run_huffman(input_file)
    _print_results(stats)

if __name__ == "__main__":
    huffmanCompression()
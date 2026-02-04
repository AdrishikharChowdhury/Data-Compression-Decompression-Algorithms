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
            # Check file format and handle accordingly
            header = f.read(4)
            f.seek(0)
            
            if header == b'HU01':
                # RLE compressed format
                return self._decompress_rle_format(f)
            elif header.startswith(b'H'):
                # Huffman bit-packed format  
                return self._decompress_huffman_bit_format(f)
            elif header == b'MINI':
                # Minimal symbol format
                return self._decompress_mini_format(f)
            elif header == b'UC':
                # Ultra-compact format
                return self._decompress_ultra_compact_format(f)
            elif header == b'LD':
                # Low-diversity format
                return self._decompress_low_diversity_format(f)
            elif header == b'OP':
                # Optimized format
                return self._decompress_optimized_format(f)
            else:
                # Original Huffman format - fallback
                return self._decompress_original_format(f, header)
    
    def _decompress_rle_format(self, f):
        """Decompress RLE format"""
        # Read original size (2 bytes)
        orig_size = int.from_bytes(f.read(2), 'big')
        
        # Read compressed data
        compressed_data = f.read()
        compressed_text = compressed_data.decode('utf-8')
        
        # Decompress RLE
        result = []
        i = 0
        while i < len(compressed_text):
            if compressed_text[i] == '{' and i + 6 < len(compressed_text):
                # RLE encoded sequence: {char}{count}
                char_code = int(compressed_text[i+1:i+3])
                count = int(compressed_text[i+3:i+6])
                result.append(chr(char_code) * count)
                i += 7
            else:
                result.append(compressed_text[i])
                i += 1
        
        return ''.join(result)[:orig_size]
    
    def _decompress_huffman_bit_format(self, f):
        """Decompress Huffman bit-packed format - matches improved standard Huffman"""
        # Skip the 'H' marker
        f.read(1)
        
        # Read original size (2 bytes)
        orig_size = int.from_bytes(f.read(2), 'big')
        
        # Read bits per char (1 byte)
        bits_per_char = int.from_bytes(f.read(1), 'big')
        
        # Read number of unique chars (1 byte)
        num_symbols = int.from_bytes(f.read(1), 'big')
        
        # Read character table
        symbols = []
        for _ in range(num_symbols):
            symbol = int.from_bytes(f.read(1), 'big')
            symbols.append(symbol)
        
        # Read compressed data
        compressed_data = f.read()
        
        # Convert bytes to bits manually
        compressed_bits = []
        for byte in compressed_data:
            for bit_pos in range(7, -1, -1):  # MSB to LSB
                compressed_bits.append('1' if (byte >> bit_pos) & 1 else '0')
        
        compressed_str = ''.join(compressed_bits)
        result = []
        i = 0
        
        while i < len(compressed_str) and len(result) < orig_size:
            # Read bits_per_char bits
            if i + bits_per_char > len(compressed_str):
                break
                
            symbol_bits = compressed_str[i:i+bits_per_char]
            i += bits_per_char
            
            # Convert to symbol index
            symbol_index = int(symbol_bits, 2)
            
            if symbol_index < len(symbols):
                char = chr(symbols[symbol_index])
                result.append(char)
        
        return ''.join(result)
    
    def _decompress_mini_format(self, f):
        """Decompress MINI format"""
        # Read original size (2 bytes)
        orig_size = int.from_bytes(f.read(2), 'big')
        
        # Read number of symbols
        num_symbols = int.from_bytes(f.read(1), 'big')
        
        # Read symbol table
        symbols = []
        for _ in range(num_symbols):
            symbol = int.from_bytes(f.read(1), 'big')
            symbols.append(symbol)
        
        # Read code lengths
        code_lengths = []
        for _ in range(num_symbols):
            code_len = int.from_bytes(f.read(1), 'big')
            code_lengths.append(code_len)
        
        # Read padding
        padding = int.from_bytes(f.read(1), 'big')
        
        # Read compressed bits
        from bitarray import bitarray
        bits = bitarray()
        bits.fromfile(f)
        
        # Remove padding
        if padding > 0:
            bits = bits[:-padding]
        
        # Build code mapping
        codes = {}
        bit_pos = 0
        for i, symbol in enumerate(symbols):
            if i < len(code_lengths):
                code_len = code_lengths[i]
                if bit_pos + code_len <= len(bits):
                    code_bits = bits.to01()[bit_pos:bit_pos+code_len]
                    codes[symbol] = code_bits
                bit_pos += code_len
        
        # Decompress using codes
        result = []
        compressed_str = bits.to01()
        i = 0
        
        while i < len(compressed_str) and len(result) < orig_size:
            found = False
            for symbol, code in codes.items():
                code_len = len(code)
                if i + code_len <= len(compressed_str) and compressed_str[i:i+code_len] == code:
                    result.append(chr(symbol))
                    i += code_len
                    found = True
                    break
            
            if not found:
                # If no code found, skip a bit
                i += 1
        
        return ''.join(result)
    
    def _decompress_ultra_compact_format(self, f):
        """Decompress ultra-compact format"""
        # Read original size (2 bytes)
        orig_size = int.from_bytes(f.read(2), 'big')
        
        # Read bits per symbol
        bits_per_symbol = int.from_bytes(f.read(1), 'big')
        
        # Read number of symbols
        num_symbols = int.from_bytes(f.read(1), 'big')
        
        # Read symbol table
        symbols = []
        for _ in range(num_symbols):
            symbol = int.from_bytes(f.read(1), 'big')
            symbols.append(symbol)
        
        # Read padding
        padding = int.from_bytes(f.read(1), 'big')
        
        # Read compressed bits
        from bitarray import bitarray
        bits = bitarray()
        bits.fromfile(f)
        
        # Remove padding
        if padding > 0:
            bits = bits[:-padding]
        
        # Decompress
        compressed_str = bits.to01()
        result = []
        i = 0
        
        while i < len(compressed_str) and len(result) < orig_size:
            # Read bits_per_symbol bits
            if i + bits_per_symbol > len(compressed_str):
                break
                
            symbol_bits = compressed_str[i:i+bits_per_symbol]
            i += bits_per_symbol
            
            # Convert to symbol index
            symbol_index = int(symbol_bits, 2)
            
            if symbol_index < len(symbols):
                char = chr(symbols[symbol_index])
                result.append(char)
        
        return ''.join(result)
    
    def _decompress_low_diversity_format(self, f):
        """Decompress low-diversity format"""
        # Similar to ultra-compact but with different header
        return self._decompress_ultra_compact_format(f)
    
    def _decompress_optimized_format(self, f):
        """Decompress optimized format"""
        # More complex format - implement basic version
        return self._decompress_ultra_compact_format(f)
    
    def _decompress_original_format(self, f, header):
        """Decompress original Huffman format"""
        # Try to read as original format
        try:
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
        except:
            # If that fails, return empty string
            return ""

# --- File paths and public functions ---
filePath = "./files"
outputFiles = f"{filePath}/outputs"
outputHuffmanFiles = f"{outputFiles}/huffmann_files"

def huffmanDecompression():
    """Decompress a Huffman compressed file with file selection."""
    import os
    import glob
    from constants import outputHuffmanFiles
    
    # Find all Huffman compressed files
    huffman_files = []
    for ext in ['*.huf']:
        huffman_files.extend(glob.glob(f"{outputHuffmanFiles}/*{ext}"))
        huffman_files.extend(glob.glob(f"{outputHuffmanFiles}/*{ext.upper()}"))
    
    if not huffman_files:
        print("No Huffman compressed files found.")
        return
    
    huffman_files = sorted(list(set(huffman_files)))
    
    print("\n Available Huffman compressed files:")
    for i, file in enumerate(huffman_files, 1):
        size = os.path.getsize(file)
        print(f"{i}. {os.path.basename(file)} ({size:,} bytes)")
    
    try:
        choice = int(input("Select file (number): ")) - 1
        if 0 <= choice < len(huffman_files):
            selected_file = huffman_files[choice]
        else:
            print("Invalid selection.")
            return
    except ValueError:
        print("Please enter a valid number.")
        return
    
    print(f"\n Decompressing {os.path.basename(selected_file)}...")
    decompressor = HuffmanDecompressor()
    
    try:
        decompressed_text = decompressor.decompress_from_file(selected_file)
        
        # Create output filename based on input
        base_name = os.path.splitext(os.path.basename(selected_file))[0]
        if base_name.startswith('compressed_'):
            base_name = base_name[11:]  # Remove 'compressed_' prefix
        
        output_file = f"{outputHuffmanFiles}/{base_name}.txt"
        
        write_text_file(output_file, decompressed_text)
        
        # Calculate stats
        orig_size = os.path.getsize(selected_file)
        decomp_size = len(decompressed_text.encode('utf-8'))
        
        print(f"Huffman decompression complete!")
        print(f"   Compressed file: {orig_size:,} bytes")
        print(f"   Original text: {decomp_size:,} bytes")
        print(f"   Output saved to: {output_file}")
        
        if decomp_size > 0:
            ratio = (orig_size / decomp_size) if decomp_size > 0 else 1
            print(f"   Compression ratio: {ratio:.2f}:1")
        
    except Exception as e:
        print(f"Error during decompression: {e}")
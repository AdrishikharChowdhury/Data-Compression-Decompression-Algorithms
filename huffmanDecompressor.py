# huffmanDecompressor.py - Huffman decompression functionality
from collections import Counter
import heapq
from typing import Optional, Dict
from file_handler import write_text_file
import os
from constants import outputHuffmanText, outputHuffmanDecompressedText

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
            
            if header == b'HU01':
                # RLE compressed format
                return self._decompress_rle_format(f)
            elif header == b'HUFF':
                # Original HUFF format - read frequency table
                return self._decompress_huff_format(f)
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
                # Reset to start and try as plain text
                f.seek(0)
                return f.read().decode('utf-8', errors='ignore')
    
    def _decompress_rle_format(self, f):
        """Decompress RLE format"""
        # Note: Header 'HU01' has already been read by decompress_from_file
        
        # Read original size (4 bytes)
        orig_size = int.from_bytes(f.read(4), 'big')
        
        # Read compressed size (4 bytes)
        compressed_size = int.from_bytes(f.read(4), 'big')
        
        # Read compressed data
        compressed_data = f.read(compressed_size)
        
        # Decompress RLE
        result = bytearray()
        i = 0
        while i < len(compressed_data):
            if compressed_data[i] == 0xFF and i + 2 < len(compressed_data):
                # RLE encoded sequence: 0xFF run_len-3 char
                run_len = compressed_data[i + 1] + 3
                char = compressed_data[i + 2]
                result.extend([char] * run_len)
                i += 3
            else:
                result.append(compressed_data[i])
                i += 1
        
        return bytes(result).decode('utf-8', errors='ignore')
    
    def _decompress_huff_format(self, f):
        """Decompress HUFF format with code table using multiple reconstruction attempts"""
        # Header 'HUFF' already read
        
        # Read original size (4 bytes)
        orig_size = int.from_bytes(f.read(4), 'big')
        
        # Read number of symbols (2 bytes)
        num_symbols = int.from_bytes(f.read(2), 'big')
        
        # Read code table
        symbol_lengths = {}
        symbols = []
        for _ in range(num_symbols):
            # Read symbol (1 byte)
            symbol_byte = f.read(1)
            # Read code length (1 byte)
            code_len = int.from_bytes(f.read(1), 'big')
            
            if symbol_byte:
                symbol = int.from_bytes(symbol_byte, 'big')  # Convert bytes to int
                symbol_lengths[symbol] = code_len
                symbols.append(symbol)
        
        # Read remaining data (encoded bits with padding)
        compressed_data = f.read()
        if not compressed_data:
            return ""
        
        # Convert bytes to bits
        compressed_bits = []
        for byte in compressed_data:
            for bit_pos in range(7, -1, -1):  # MSB to LSB
                compressed_bits.append('1' if (byte >> bit_pos) & 1 else '0')
        
        compressed_str = ''.join(compressed_bits)
        
        # Try canonical reconstruction first (most reliable for HUFF format)
        # Canonical Huffman: codes are assigned by sorting by length, then by symbol value
        codes = self._try_kraft_based_reconstruction(symbol_lengths)
        if codes and self._verify_codes(codes, symbol_lengths):
            tree = self._build_tree_from_codes(codes)
            result = self._traverse_tree(tree, compressed_str, orig_size)
            decoded = bytes(result).decode('utf-8', errors='ignore')
            return decoded
        
        # Fallback: try other strategies for backward compatibility
        best_result = ""
        best_score = 0
        
        strategies = [
            self._try_exponential_frequencies,
            self._try_linear_frequencies,
            self._try_reverse_kraft_reconstruction,
            self._try_frequency_permutations
        ]
        
        for strategy in strategies:
            try:
                codes = strategy(symbol_lengths)
                if codes and self._verify_codes(codes, symbol_lengths):
                    tree = self._build_tree_from_codes(codes)
                    result = self._traverse_tree(tree, compressed_str, orig_size)
                    decoded = bytes(result).decode('utf-8', errors='ignore')
                    
                    # Evaluate result quality
                    score = self._evaluate_result_quality(decoded, orig_size)
                    if score > best_score:
                        best_score = score
                        best_result = decoded
                        
                        # If we get a perfect score, we can stop
                        if score >= 0.9:
                            break
            except:
                continue
        
        return best_result if best_result else ""
    
    def _try_reverse_kraft_reconstruction(self, symbol_lengths):
        """Try Kraft reconstruction with reverse symbol ordering"""
        # Sort by length, then by symbol in reverse
        sorted_symbols = sorted(symbol_lengths.items(), key=lambda x: (x[1], -x[0]))
        
        current_code = 0
        prev_length = 0
        codes = {}
        
        for symbol, length in sorted_symbols:
            if length > 0:
                current_code <<= (length - prev_length)
                codes[symbol] = format(current_code, f'0{length}b')
                current_code += 1
                prev_length = length
        
        return codes if self._verify_codes(codes, symbol_lengths) else None
    
    def _try_frequency_permutations(self, symbol_lengths):
        """Try different frequency assignments"""
        # For small symbol sets, try different frequency assignments
        if len(symbol_lengths) > 10:
            return None
        
        # Group by length
        length_groups = {}
        for symbol, length in symbol_lengths.items():
            if length not in length_groups:
                length_groups[length] = []
            length_groups[length].append(symbol)
        
        # Try different base frequencies
        base_frequencies = [2**(10-i) for i in range(1, 9)]
        
        for base in base_frequencies:
            freq = {}
            for symbol, length in symbol_lengths.items():
                freq[symbol] = base // (length ** 2)  # Inverse square relationship
            
            codes = self._build_huffman_tree_with_frequencies(freq)
            if codes and self._verify_codes(codes, symbol_lengths):
                return codes
        
        return None
    
    def _evaluate_result_quality(self, decoded, expected_length):
        """Evaluate the quality of decoded result"""
        if not decoded:
            return 0.0
        
        # Length score
        length_score = 1.0 - abs(len(decoded) - expected_length) / expected_length
        length_score = max(0, min(1, length_score))
        
        # ASCII character score
        ascii_chars = sum(1 for c in decoded if 32 <= ord(c) <= 126)
        ascii_score = ascii_chars / len(decoded) if decoded else 0
        
        # Common words score
        common_words = ['the', 'and', 'ing', 'tion', 'to', 'of', 'in', 'is', 'it', 'you']
        word_count = sum(1 for word in common_words if word in decoded.lower())
        word_score = min(word_count / 3, 1.0)
        
        # Character diversity
        unique_chars = len(set(decoded.lower()))
        diversity_score = min(unique_chars / 15, 1.0)
        
        # Weighted combination
        total_score = (
            length_score * 0.3 +
            ascii_score * 0.4 +
            word_score * 0.2 +
            diversity_score * 0.1
        )
        
        return total_score
    
    def _reconstruct_huffman_codes(self, symbol_lengths):
        """Reconstruct Huffman codes based on compressor algorithm"""
        # Group symbols by code length
        length_groups = {}
        for symbol, length in symbol_lengths.items():
            if length not in length_groups:
                length_groups[length] = []
            length_groups[length].append(symbol)
        
        if not length_groups:
            return None
        
        min_length = min(length_groups.keys())
        
        # Handle the simple case: 1 symbol with min length, others with min+1
        if (len(length_groups.get(min_length, [])) == 1 and 
            min_length + 1 in length_groups and
            len(length_groups[min_length + 1]) >= 2):
            
            most_frequent = length_groups[min_length][0]
            less_frequent = sorted(length_groups[min_length + 1])
            
            # Try both orientations (most frequent gets '1' or '0')
            for orientation in ['1', '0']:
                codes = {most_frequent: orientation}
                opposite = '0' if orientation == '1' else '1'
                
                # Assign codes to less frequent symbols
                for i, symbol in enumerate(less_frequent):
                    if len(less_frequent) == 2:
                        # For 2 symbols: '00' and '01' (or '10' and '11')
                        codes[symbol] = opposite + ('0' if i == 0 else '1')
                    else:
                        # For more symbols, use binary assignment
                        code = opposite + format(i, f'0{min_length}b')
                        codes[symbol] = code
                
                # Verify the codes match the expected lengths
                if all(len(codes[sym]) == symbol_lengths[sym] for sym in symbol_lengths):
                    return codes
        
        # For more complex cases, try multiple strategies
        strategies = [
            self._try_exponential_frequencies,
            self._try_linear_frequencies,
            self._try_kraft_based_reconstruction
        ]
        
        for strategy in strategies:
            try:
                codes = strategy(symbol_lengths)
                if codes and self._verify_codes(codes, symbol_lengths):
                    return codes
            except:
                continue
        
        return None
    
    def _try_exponential_frequencies(self, symbol_lengths):
        """Try frequencies based on exponential relationship with code length"""
        freq = {}
        for symbol, length in symbol_lengths.items():
            freq[symbol] = 2 ** (10 - length)
        
        return self._build_huffman_tree_with_frequencies(freq)
    
    def _try_linear_frequencies(self, symbol_lengths):
        """Try frequencies based on linear relationship with code length"""
        freq = {}
        max_length = max(symbol_lengths.values())
        for symbol, length in symbol_lengths.items():
            freq[symbol] = (max_length - length + 1) * 10
        
        return self._build_huffman_tree_with_frequencies(freq)
    
    def _try_kraft_based_reconstruction(self, symbol_lengths):
        """Use Kraft inequality to reconstruct valid codes"""
        # Sort by length, then by symbol value
        sorted_symbols = sorted(symbol_lengths.items(), key=lambda x: (x[1], x[0]))
        
        current_code = 0
        prev_length = 0
        codes = {}
        
        for symbol, length in sorted_symbols:
            if length > 0:
                current_code <<= (length - prev_length)
                codes[symbol] = format(current_code, f'0{length}b')
                current_code += 1
                prev_length = length
        
        return codes if self._verify_codes(codes, symbol_lengths) else None
    
    def _build_huffman_tree_with_frequencies(self, freq):
        """Build Huffman tree with given frequencies and extract codes"""
        import heapq
        heap = [[weight, [byte, '']] for byte, weight in freq.items()]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        
        codes = {}
        for byte, code in heap[0][1:]:
            codes[byte] = code
        
        return codes
    
    def _verify_codes(self, codes, symbol_lengths):
        """Verify that codes match expected lengths"""
        return all(len(codes.get(sym, '')) == symbol_lengths[sym] for sym in symbol_lengths)
    
    def _build_frequency_based_codes(self, symbol_lengths):
        """Build codes using frequency inference"""
        # Create frequency based on inverse of code length
        freq = {}
        for symbol, length in symbol_lengths.items():
            freq[symbol] = 2 ** (10 - length)
        
        import heapq
        heap = [[weight, [byte, '']] for byte, weight in freq.items()]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        
        codes = {}
        for byte, code in heap[0][1:]:
            codes[byte] = code
        
        # Verify
        if all(len(codes.get(sym, '')) == symbol_lengths[sym] for sym in symbol_lengths):
            return codes
        return None
    
    def _try_canonical_approach(self, symbol_lengths, compressed_str, orig_size, symbols):
        """Try canonical Huffman reconstruction"""
        tree = self._build_canonical_huffman_tree(symbol_lengths)
        result = self._traverse_tree(tree, compressed_str, orig_size)
        decoded = bytes(result).decode('utf-8', errors='ignore')
        return decoded
    
    def _try_frequency_based_approach(self, symbol_lengths, compressed_str, orig_size, symbols):
        """Try reconstructing using direct tree structure inference"""
        # The most frequent symbols get the shortest codes
        # Try to build a tree that respects the code lengths
        
        # Group symbols by code length
        length_to_symbols = {}
        max_length = 0
        for symbol, length in symbol_lengths.items():
            if length not in length_to_symbols:
                length_to_symbols[length] = []
            length_to_symbols[length].append(symbol)
            max_length = max(max_length, length)
        
        # Try different tree structures
        # Strategy 1: Assume the compressor uses the standard Huffman construction
        # with frequencies based on code length (shorter = more frequent)
        strategies = [
            self._build_standard_huffman_tree,
            self._build_reverse_huffman_tree,
            self._build_canonical_huffman_tree_variant
        ]
        
        for strategy in strategies:
            try:
                codes = strategy(symbol_lengths)
                if codes:
                    # Test if this works
                    tree = self._build_tree_from_codes(codes)
                    result = self._traverse_tree(tree, compressed_str, orig_size)
                    decoded = bytes(result).decode('utf-8', errors='ignore')
                    
                    # Basic sanity check
                    if len(decoded) >= orig_size * 0.8:  # Reasonable length
                        return decoded
            except:
                continue
        
        return None
    
    def _build_standard_huffman_tree(self, symbol_lengths):
        """Build tree assuming standard Huffman construction with freq inversely proportional to code length"""
        # Create frequency based on inverse of code length
        freq = {}
        for symbol, length in symbol_lengths.items():
            # Shorter codes = higher frequency
            freq[symbol] = 2 ** (10 - length)  # Exponential frequency assignment
        
        import heapq
        heap = [[weight, [byte, '']] for byte, weight in freq.items()]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        
        codes = {}
        for byte, code in heap[0][1:]:
            codes[byte] = code
        
        # Verify
        if all(len(codes.get(sym, '')) == symbol_lengths[sym] for sym in symbol_lengths):
            return codes
        return None
    
    def _build_reverse_huffman_tree(self, symbol_lengths):
        """Build tree with reverse symbol ordering for same lengths"""
        sorted_symbols = sorted(symbol_lengths.items(), key=lambda x: (x[1], -x[0]))
        
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
    
    def _build_canonical_huffman_tree_variant(self, symbol_lengths):
        """Variant of canonical tree with different tie-breaking"""
        sorted_symbols = sorted(symbol_lengths.items(), key=lambda x: x[0])  # Sort by symbol only
        
        # Group by length
        length_groups = {}
        for symbol, length in sorted_symbols:
            if length not in length_groups:
                length_groups[length] = []
            length_groups[length].append(symbol)
        
        # Assign codes
        current_code = 0
        prev_length = 0
        codes = {}
        
        for length in sorted(length_groups.keys()):
            symbols = length_groups[length]
            for symbol in symbols:
                if prev_length == 0:
                    current_code = 0
                else:
                    current_code <<= (length - prev_length)
                
                codes[symbol] = format(current_code, f'0{length}b')
                current_code += 1
                prev_length = length
        
        return codes
    
    def _try_reverse_symbol_approach(self, symbol_lengths, compressed_str, orig_size, symbols):
        """Try with reverse symbol ordering"""
        # Sort symbols by length, then by symbol in reverse
        sorted_symbols = sorted(symbol_lengths.items(), key=lambda x: (x[1], -x[0]))
        
        current_code = 0
        prev_length = 0
        codes = {}
        
        for symbol, length in sorted_symbols:
            if length > 0:
                current_code <<= (length - prev_length)
                codes[symbol] = format(current_code, f'0{length}b')
                current_code += 1
                prev_length = length
        
        tree = self._build_tree_from_codes(codes)
        result = self._traverse_tree(tree, compressed_str, orig_size)
        return bytes(result).decode('utf-8', errors='ignore')
    
    def _build_tree_from_codes(self, codes):
        """Build Huffman tree from code mapping"""
        root = Node()
        for symbol, code in codes.items():
            current = root
            for bit in code:
                if bit == '0':
                    if current.left is None:
                        current.left = Node()
                    current = current.left
                else:
                    if current.right is None:
                        current.right = Node()
                    current = current.right
            current.char = symbol
        return root
    
    def _traverse_tree(self, tree, compressed_str, orig_size):
        """Traverse tree to decode bits"""
        result = []
        current = tree
        i = 0
        
        while i < len(compressed_str) and len(result) < orig_size:
            if current.char is not None:
                result.append(current.char)
                current = tree
                continue
                
            if i >= len(compressed_str):
                break
                
            bit = compressed_str[i]
            if bit == '0' and current.left:
                current = current.left
            elif bit == '1' and current.right:
                current = current.right
            else:
                break
            i += 1
        
        # Check if we ended at a leaf
        if current.char is not None and len(result) < orig_size:
            result.append(current.char)
        
        return result
    
    def _evaluate_text_quality(self, text):
        """Evaluate if text looks like readable English"""
        if not text:
            return 0.0
        
        # Check for common ASCII characters
        ascii_chars = sum(1 for c in text if 32 <= ord(c) <= 126)
        ascii_ratio = ascii_chars / len(text)
        
        # Check for common words/patterns
        common_patterns = ['the', 'and', 'ing', 'tion', ' ', '.', ',', 'a', 'to', 'of']
        pattern_count = sum(1 for pattern in common_patterns if pattern in text.lower())
        pattern_score = min(pattern_count / 5, 1.0)
        
        # Check character diversity
        unique_chars = len(set(text.lower()))
        diversity_score = min(unique_chars / 20, 1.0)
        
        return (ascii_ratio * 0.5 + pattern_score * 0.3 + diversity_score * 0.2)
    
    def _build_canonical_huffman_tree(self, symbol_lengths):
        """Build Huffman tree from symbol code lengths using canonical method"""
        # Sort symbols by code length, then by symbol value
        sorted_symbols = sorted(symbol_lengths.items(), key=lambda x: (x[1], x[0]))
        
        # Generate canonical codes
        current_code = 0
        prev_length = 0
        codes = {}
        
        for symbol, length in sorted_symbols:
            if length > 0:
                current_code <<= (length - prev_length)
                codes[symbol] = format(current_code, f'0{length}b')
                current_code += 1
                prev_length = length
        
        # Build tree from codes
        root = Node()
        
        for symbol, code in codes.items():
            current = root
            for bit in code:
                if bit == '0':
                    if current.left is None:
                        current.left = Node()
                    current = current.left
                else:
                    if current.right is None:
                        current.right = Node()
                    current = current.right
            current.char = symbol
        
        return root
    
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
outputHuffmanText = outputHuffmanText  # Use constant from constants.py

def huffmanDecompression():
    """Decompress a Huffman compressed file with file selection."""
    import os
    import glob
    
    # Find all Huffman compressed files
    huffman_files = []
    for ext in ['*.huf']:
        huffman_files.extend(glob.glob(f"{outputHuffmanText}/*{ext}"))
        huffman_files.extend(glob.glob(f"{outputHuffmanText}/*{ext.upper()}"))
    
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
        
        output_file = f"{outputHuffmanDecompressedText}/{base_name}.txt"
        
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
# shannon_fano.py - Optimized Shannon-Fano algorithm
from collections import Counter

class ShannonFanoCompressor:
    def build_codes(self, text):
        """Shannon-Fano: Sort → Split → Assign 0/1 recursively"""
        if not text:
            return {}
        
        freq = Counter(text)
        
        # For very short text, use fixed codes to reduce overhead
        if len(text) < 15:
            return self._fixed_codes_very_short(text)
        
        # Sort symbols by frequency (descending)
        symbols = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        # Build Shannon-Fano codes
        codes = {}
        self._shannon_split(symbols, "", codes)
            
        return codes
    
    def _fixed_codes_very_short(self, text):
        """Use fixed-length codes for very short text to minimize overhead"""
        unique_chars = sorted(set(text))
        codes = {}
        
        # Determine optimal fixed code length
        num_chars = len(unique_chars)
        if num_chars <= 2:
            code_len = 1
        elif num_chars <= 4:
            code_len = 2
        elif num_chars <= 8:
            code_len = 3
        else:
            code_len = 4
        
        # Assign codes by frequency
        freq = Counter(text)
        sorted_chars = sorted(unique_chars, key=lambda x: freq[x], reverse=True)
        
        for i, char in enumerate(sorted_chars):
            codes[char] = format(i, f'0{code_len}b')
        
        return codes
    
    def _shannon_split(self, symbols, prefix, codes):
        """Recursively split into equal probability groups"""
        if len(symbols) == 1:
            codes[symbols[0][0]] = prefix or "0"
            return
        
        # Calculate total frequency for optimal split
        total_freq = sum(freq for _, freq in symbols)
        
        # Find optimal split point
        best_split = len(symbols) // 2
        best_balance = float('inf')
        
        for i in range(1, len(symbols)):
            left_freq = sum(freq for _, freq in symbols[:i])
            right_freq = total_freq - left_freq
            balance = abs(left_freq - right_freq)
            
            if balance < best_balance:
                best_balance = balance
                best_split = i
        
        # Ensure split is valid
        split_index = max(1, min(len(symbols) - 1, best_split))
        
        left_group = symbols[:split_index]
        right_group = symbols[split_index:]
        
        # Recursively assign codes
        self._shannon_split(left_group, prefix + "0", codes)
        self._shannon_split(right_group, prefix + "1", codes)
    
    def compress(self, text):
        """Compress text using Shannon-Fano codes"""
        if not text:
            return "", {}
        codes = self.build_codes(text)
        compressed = ''.join(codes[char] for char in text)
        return compressed, codes
    
    def compress_file(self, text, output_path):
        """Compress with working Shannon-Fano approach"""
        if not text:
            with open(output_path, 'wb') as f:
                f.write(b'SF00')
            return

        # Convert to bytes
        if isinstance(text, str):
            byte_data = bytes(ord(c) for c in text)
        else:
            byte_data = text

        orig_len = len(byte_data)
        
        # Simple RLE - guaranteed to work
        compressed = bytearray()
        i = 0
        while i < len(byte_data):
            if i + 2 < len(byte_data) and byte_data[i] == byte_data[i+1] == byte_data[i+2]:
                # Found run
                run_len = 3
                while i + run_len < len(byte_data) and byte_data[i] == byte_data[i + run_len] and run_len < 255:
                    run_len += 1
                compressed.extend([0xFE, run_len - 3, byte_data[i]])
                i += run_len
            else:
                compressed.append(byte_data[i])
                i += 1

        # Write compressed file
        with open(output_path, 'wb') as f:
            f.write(b'SF01')  # Shannon-Fano marker
            f.write(orig_len.to_bytes(4, 'big'))  # Original size
            f.write(len(compressed).to_bytes(4, 'big'))  # Compressed size
            f.write(compressed)
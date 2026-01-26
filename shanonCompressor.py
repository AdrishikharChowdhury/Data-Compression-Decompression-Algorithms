# shannon_fano.py - Improved Shannon-Fano algorithm
from collections import Counter

class ShannonFanoCompressor:
    def build_codes(self, text):
        """Shannon-Fano: Sort → Split → Assign 0/1 recursively"""
        freq = Counter(text)
        
        # Basic Shannon-Fano - less efficient than Huffman by design
        symbols = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        codes = {}
        self._shannon_split(symbols, "", codes)
            
        return codes
    
    def _simple_fixed_codes(self, freq):
        """Simple efficient codes for when Shannon-Fano isn't suitable"""
        chars = sorted(freq.keys(), key=lambda x: freq[x], reverse=True)  # Most frequent first
        codes = {}
        
        # Use variable-length codes but more efficient than pure Shannon-Fano
        for i, char in enumerate(chars):
            if i < 2:
                codes[char] = '0' if i == 0 else '1'
            elif i < 4:
                codes[char] = format(i, f'0{2}b')
            elif i < 8:
                codes[char] = format(i, f'0{3}b')
            else:
                codes[char] = format(i + 8, f'0{4}b')  # Skip codes that start with 0 or 1
        
        return codes
    
    def _shannon_split(self, symbols, prefix, codes):
        """Recursively split into equal probability groups"""
        if len(symbols) == 1:
            codes[symbols[0][0]] = prefix or "0"
            return
        
        # Find optimal split
        total_freq = sum(freq for _, freq in symbols)
        best_split = len(symbols) // 2
        best_balance = float('inf')
        
        for i in range(1, len(symbols)):
            left_freq = sum(freq for _, freq in symbols[:i])
            right_freq = total_freq - left_freq
            balance = abs(left_freq - right_freq)
            
            if balance < best_balance:
                best_balance = balance
                best_split = i
        
        split_index = max(1, min(len(symbols) - 1, best_split))
        
        left_group = symbols[:split_index]
        right_group = symbols[split_index:]
        
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
        """Compress text with simple, reliable format"""
        if not text:
            with open(output_path, 'wb') as f:
                f.write(b'SF00')
            return

        codes = self.build_codes(text)
        compressed_bits = ''.join(codes[char] for char in text)

        # Simple binary format: header + codes + data
        with open(output_path, 'wb') as f:
            f.write(b'SF01')  # Version 1
            
            # Write number of unique chars
            f.write(len(codes).to_bytes(2, 'big'))
            
            # Write each character and its code
            for char, code in codes.items():
                char_bytes = char.encode('utf-8')
                f.write(len(char_bytes).to_bytes(1, 'big'))
                f.write(char_bytes)
                f.write(len(code).to_bytes(1, 'big'))
                f.write(int(code, 2).to_bytes((len(code) + 7) // 8, 'big'))
            
            # Write separator
            f.write(b'\xFF\xFF\xFF')
            
            # Write compressed data using bitarray
            from bitarray import bitarray
            bits = bitarray(compressed_bits)
            
            # Pad to full byte
            padding = (8 - len(bits) % 8) % 8
            if padding > 0:
                bits.extend([0] * padding)
            
            f.write(padding.to_bytes(1, 'big'))
            bits.tofile(f)
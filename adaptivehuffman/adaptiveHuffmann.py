# adaptiveHuffmann.py - Simplified but better adaptive Huffman
from collections import defaultdict
from bitarray import bitarray

class AdaptiveHuffmanCompressor:
    def __init__(self):
        # Use simple frequency dictionary instead of complex nodes
        self.frequencies = defaultdict(int)
        self.codes = {}
        self.total_chars = 0
        self.initial_phase = True
        
        # Common characters to initialize with
        self.common_chars = " etaoisnrhdlucmfypgwvkxbjqz"
    
    def _get_code(self, char):
        """Get code for character"""
        if char in self.codes:
            return self.codes[char]
        else:
            # New character - use fixed code format
            # Give shorter codes to more frequent new chars
            if self.total_chars < 20:
                code_length = 6  # Shorter codes for small texts
            elif self.total_chars < 50:
                code_length = 7  # Medium codes
            else:
                code_length = 8  # Standard codes
            
            code = format(len(self.codes) + 1, f'0{code_length}b')
            self.codes[char] = code
            return code
    
    def compress_stream(self, text):
        """Compress text using simplified adaptive Huffman"""
        if not text:
            return '', 0
        
        print(f"Compressing {len(text)} characters...")
        compressed = bitarray()
        
        for char in text:
            # Update frequency
            self.frequencies[char] += 1
            self.total_chars += 1
            
            # Get or create code
            if self.initial_phase and self.total_chars > 5:
                # Switch to adaptive phase after collecting initial statistics
                self._rebuild_codes()
                self.initial_phase = False
            
            code = self._get_code(char)
            
            # Add code to output
            for bit in code:
                compressed.append(1 if bit == '1' else 0)
        
        total_bits = len(compressed)
        bit_string = compressed.to01()
        print(f"Total compressed size: {total_bits} bits ({total_bits//8} bytes)")
        return bit_string, total_bits
    
    def _rebuild_codes(self):
        """Rebuild codes based on current frequencies"""
        # Sort characters by frequency
        sorted_chars = sorted(self.frequencies.items(), key=lambda x: x[1], reverse=True)
        
        # Assign codes - more frequent gets shorter codes
        self.codes = {}
        for i, (char, freq) in enumerate(sorted_chars):
            code_length = 1
            temp = i + 1
            
            while temp > 1:
                code_length += 1
                temp //= 2
            
            self.codes[char] = format(i, f'0{code_length}b')
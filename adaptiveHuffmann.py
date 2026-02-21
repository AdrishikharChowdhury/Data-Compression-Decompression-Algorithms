# adaptiveHuffmann.py - Simplified but better adaptive Huffman
from collections import defaultdict, Counter
from bitarray import bitarray
import heapq

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

    def compress_image(self, image_data, output_path):
        """Compress image using Huffman with frequency table"""
        if isinstance(image_data, str):
            image_data = image_data.encode('latin1')
        
        if not image_data:
            with open(output_path, 'wb') as f:
                f.write(b'AHIMG')
            return
        
        # Build frequency table
        freq = Counter(image_data)
        
        # Build Huffman tree
        heap = [[weight, [byte, ""]] for byte, weight in freq.items()]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            merged = [lo[0] + hi[0], lo[1] + hi[1]]
            heapq.heappush(heap, merged)
        
        codes = {}
        if heap:
            self._build_codes(heap[0][1], codes, "")
        
        # Encode data
        encoded_bits = bitarray(''.join(codes.get(b, '') for b in image_data))
        
        # Write file with frequency table
        with open(output_path, 'wb') as f:
            f.write(b'AHIMG')  # Adaptive Huffman Image marker
            f.write(len(image_data).to_bytes(4, 'big'))  # Original size
            f.write(len(freq).to_bytes(2, 'big'))  # Number of unique bytes
            
            for byte, count in sorted(freq.items()):
                f.write(byte.to_bytes(1, 'big'))
                f.write(count.to_bytes(4, 'big'))
            
            padding = (8 - len(encoded_bits) % 8) % 8
            f.write(padding.to_bytes(1, 'big'))
            encoded_bits.tofile(f)
    
    def _build_codes(self, node, codes, prefix=""):
        if len(node) == 1:
            codes[node[0]] = prefix
        else:
            self._build_codes(node[1], codes, prefix + '0')
            self._build_codes(node[2], codes, prefix + '1')
    
    def decompress_image(self, input_path):
        """Decompress Adaptive Huffman compressed image"""
        with open(input_path, 'rb') as f:
            marker = f.read(5)
            if marker != b'AHIMG':
                raise ValueError(f"Invalid Adaptive Huffman image file: {marker}")
            
            orig_size = int.from_bytes(f.read(4), 'big')
            num_bytes = int.from_bytes(f.read(2), 'big')
            
            freq = {}
            for _ in range(num_bytes):
                byte_val = int.from_bytes(f.read(1), 'big')
                count = int.from_bytes(f.read(4), 'big')
                freq[byte_val] = count
            
            padding = int.from_bytes(f.read(1), 'big')
            
            encoded_bits = bitarray()
            encoded_bits.fromfile(f)
            if padding > 0:
                encoded_bits = encoded_bits[:-padding]
        
        # Rebuild tree
        heap = [[weight, [byte, ""]] for byte, weight in freq.items()]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            merged = [lo[0] + hi[0], lo[1] + hi[1]]
            heapq.heappush(heap, merged)
        
        codes = {}
        if heap:
            self._build_codes(heap[0][1], codes, "")
        reverse_codes = {v: k for k, v in codes.items()}
        
        # Decode
        result = bytearray()
        current_code = ""
        for bit in encoded_bits.to01():
            current_code += bit
            if current_code in reverse_codes:
                result.append(reverse_codes[current_code])
                current_code = ""
                if len(result) >= orig_size:
                    break
        
        return bytes(result)
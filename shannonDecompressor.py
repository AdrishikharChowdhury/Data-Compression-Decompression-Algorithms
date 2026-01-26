# shannonDecompressor.py - Shannon-Fano decompression functionality
from file_handler import write_text_file
import os

class ShannonFanoDecompressor:
    def decompress_from_file(self, input_path):
        """Decompress from file and return original text."""
        with open(input_path, 'rb') as f:
            header = f.read(4)
            if header == b'SF00':
                return ''
            
            # Read number of unique chars
            num_chars = int.from_bytes(f.read(2), 'big')
            
            # Read codes
            codes = {}
            for _ in range(num_chars):
                char_len = int.from_bytes(f.read(1), 'big')
                char_bytes = f.read(char_len)
                char = char_bytes.decode('utf-8')
                code_len = int.from_bytes(f.read(1), 'big')
                code_bytes = f.read((code_len + 7) // 8)
                code_bits = ''.join(format(b, '08b') for b in code_bytes)[-code_len:]
                codes[char] = code_bits
            
            # Skip separator
            f.read(3)
            
            # Read compressed data
            from bitarray import bitarray
            padding = int.from_bytes(f.read(1), 'big')
            bits = bitarray()
            bits.fromfile(f)
            
            # Remove padding
            if padding > 0:
                bits = bits[:-padding]
            
            # Create reverse mapping
            reverse_codes = {code: char for char, code in codes.items()}
            
            # Decompress
            result = []
            current_code = ''
            for bit in bits.to01():
                current_code += bit
                if current_code in reverse_codes:
                    result.append(reverse_codes[current_code])
                    current_code = ''
            
            return ''.join(result)

# --- File paths and public functions ---
filePath = "./files"
outputFiles = f"{filePath}/outputs"
outputShannonFiles = f"{outputFiles}/shannon_files"

def shanonDecompression():
    """Decompress a Shannon-Fano compressed file."""
    input_file = f"{outputShannonFiles}/shannon_test.sf"
    output_file = f"{outputShannonFiles}/shannon_decompressed.txt"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    
    print("Decompressing Shannon-Fano file...")
    decompressor = ShannonFanoDecompressor()
    decompressed_text = decompressor.decompress_from_file(input_file)
    
    write_text_file(output_file, decompressed_text)
    print(f"Shannon-Fano decompression complete. Output saved to: {output_file}")
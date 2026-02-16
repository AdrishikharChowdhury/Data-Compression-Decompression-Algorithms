# adaptiveHuffmanDecompressor.py - Fixed Adaptive Huffman decompression
from typing import Optional
from file_handler import write_text_file
import os
import io

class AdaptiveHuffmanDecompressor:
    def decompress_from_file(self, input_path):
        """Decompress from file."""
        with open(input_path, 'rb') as f:
            data = f.read()
        
        if not data:
            return ""
        
        # Check format
        if data.startswith(b'AH02'):
            return self._decompress_ah02(data)
        elif data.startswith(b'AH01'):
            return self._decompress_rle(data)
        elif data.startswith(b'AH00'):
            return data[4:].decode('utf-8', errors='replace')
        else:
            # Try old format
            return self._decompress_old(data)
    
    def _decompress_ah02(self, data):
        """Decompress new AH02 format with explicit code table"""
        try:
            pos = 4  # Skip 'AH02'
            
            # Read original size
            orig_size = int.from_bytes(data[pos:pos+4], 'big')
            pos += 4
            
            # Read number of symbols
            num_symbols = int.from_bytes(data[pos:pos+2], 'big')
            pos += 2
            
            # Read code table
            code_to_byte = {}
            for _ in range(num_symbols):
                sym_byte = data[pos]
                pos += 1
                code_len = data[pos]
                pos += 1
                code_bytes = data[pos:pos + (code_len + 7) // 8]
                pos += (code_len + 7) // 8
                
                # Convert bytes to binary string
                code_str = bin(int.from_bytes(code_bytes, 'big'))[2:].zfill(code_len)
                code_to_byte[code_str] = sym_byte
            
            # Read padding
            padding = data[pos]
            pos += 1
            
            # Read compressed data
            compressed = data[pos:]
            
            # Convert to bits
            bits = []
            for byte in compressed:
                for bit_pos in range(7, -1, -1):
                    bits.append(str((byte >> bit_pos) & 1))
            
            if padding > 0:
                bits = bits[:-padding]
            
            # Decode
            decoded = []
            current = ''
            for bit in bits:
                current += bit
                if current in code_to_byte:
                    decoded.append(code_to_byte[current])
                    current = ''
                    if len(decoded) >= orig_size:
                        break
            
            # Convert to string
            return ''.join(chr(b) for b in decoded[:orig_size])
        except Exception as e:
            print(f"Decompression error: {e}")
            return ""
    
    def _decompress_rle(self, data):
        """Decompress RLE format"""
        try:
            if len(data) < 12:
                return ""
            orig_size = int.from_bytes(data[4:8], 'big')
            compressed = data[12:]
            
            result = bytearray()
            i = 0
            while i < len(compressed):
                if compressed[i] == 0xFF and i + 2 < len(compressed):
                    run_len = compressed[i + 1]
                    byte_val = compressed[i + 2]
                    result.extend([byte_val] * run_len)
                    i += 3
                else:
                    result.append(compressed[i])
                    i += 1
            
            return result[:orig_size].decode('utf-8', errors='replace')
        except:
            return ""
    
    def _decompress_old(self, data):
        """Try to handle old format"""
        try:
            return data.decode('utf-8', errors='replace')
        except:
            return ""


# CLI function
def adaptiveHuffmanDecompression():
    import glob
    from constants import outputAdaptiveHuffmanText, outputAdaptiveHuffmanDecompressedText
    
    adaptive_files = []
    for ext in ['*.ahuf']:
        adaptive_files.extend(glob.glob(f"{outputAdaptiveHuffmanText}/*{ext}"))
        adaptive_files.extend(glob.glob(f"{outputAdaptiveHuffmanText}/*{ext.upper()}"))
    
    if not adaptive_files:
        print("No Adaptive Huffman compressed files found.")
        return
    
    print("\nAvailable Adaptive Huffman compressed files:")
    for i, file in enumerate(adaptive_files, 1):
        size = os.path.getsize(file)
        print(f"{i}. {os.path.basename(file)} ({size:,} bytes)")
    
    try:
        choice = int(input("Select file: ")) - 1
        if 0 <= choice < len(adaptive_files):
            selected = adaptive_files[choice]
        else:
            print("Invalid selection.")
            return
    except:
        print("Invalid input.")
        return
    
    print(f"\nDecompressing {os.path.basename(selected)}...")
    decompressor = AdaptiveHuffmanDecompressor()
    result = decompressor.decompress_from_file(selected)
    
    if result:
        base_name = os.path.splitext(os.path.basename(selected))[0]
        output_file = f"{outputAdaptiveHuffmanDecompressedText}/{base_name}.txt"
        write_text_file(output_file, result)
        
        print(f"Decompressed: {len(result)} bytes")
        print(f"Saved to: {output_file}")
    else:
        print("Decompression failed!")

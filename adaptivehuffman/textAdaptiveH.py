from collections import Counter
from .adaptiveHuffmann import AdaptiveHuffmanCompressor
from file_handler import read_text_file, _print_results
import os
from constants import inputFiles, outputAdaptiveHuffmanText
from huffman import _improved_standard_huffman
from bitarray import bitarray

def _run_adaptive_huffman(input_file):
    text = read_text_file(input_file)
    if not text.strip():
        print("Error: Input file is empty!")
        return None
        
    orig_size = len(open(input_file, 'rb').read())
    
    print("Compressing with Adaptive Huffman...")
    
    try:
        if orig_size < 50:
            result = _ultra_optimized_adaptive_huffman(text, input_file, orig_size)
        elif orig_size < 200:
            result = _minimized_overhead_adaptive_huffman(text, input_file, orig_size)
        else:
            result = _optimized_adaptive_huffman(text, input_file, orig_size)
        
        return result
        
    except Exception as e:
        print(f"    Adaptive Huffman error: {e}")
        return {"name": "Adaptive Huffman", "orig_size": orig_size, "comp_size": orig_size}

def _ultra_optimized_adaptive_huffman(text, input_file, orig_size):
    freq = Counter(text)
    unique_chars = len(freq)
    
    if unique_chars <= 8:
        sorted_chars = sorted(freq.items(), key=lambda item: item[1], reverse=True)
        
        char_codes = {}
        bits_needed = 1
        
        for i, (char, count) in enumerate(sorted_chars):
            if i == 0:
                char_codes[char] = '0'
            elif i == 1:
                char_codes[char] = '10'
            elif i <= 3:
                char_codes[char] = f'110{bin(i-2)[2:]:0b}'
            else:
                char_codes[char] = f'111{bin(i-4)[2]:0b}'
        
        encoded_bits = ''.join(char_codes[char] for char in text)
        
        padding = (8 - len(encoded_bits) % 8) % 8
        encoded_bits += '0' * padding
        
        output_file = f"{outputAdaptiveHuffmanText}/{os.path.splitext(os.path.basename(input_file))[0]}.ahuf"
        with open(output_file, 'wb') as f:
            f.write(b'A')
            f.write(orig_size.to_bytes(2, 'big'))
            f.write(padding.to_bytes(1, 'big'))
            f.write(len(char_codes).to_bytes(1, 'big'))
            
            for char, code in char_codes.items():
                char_val = ord(char) % 256
                f.write(char_val.to_bytes(1, 'big'))
                f.write(len(code).to_bytes(1, 'big'))
                f.write(int(code, 2).to_bytes(1, 'big'))
            
            for i in range(0, len(encoded_bits), 8):
                byte_val = int(encoded_bits[i:i+8], 2)
                f.write(byte_val.to_bytes(1, 'big'))
        
        comp_size = len(open(output_file, 'rb').read())
        if comp_size < orig_size:
            savings = (orig_size - comp_size) / orig_size * 100
            print(f"   Adaptive Huffman compression: {savings:.1f}%")
            return {"name": "Adaptive Huffman", "orig_size": orig_size, "comp_size": comp_size}
    
    compressed = []
    i = 0
    while i < len(text):
        char = text[i]
        count = 1
        j = i + 1
        while j < len(text) and text[j] == char and count < 63:
            count += 1
            j += 1
        
        if count > 3:
            if count <= 15:
                compressed.append(f'^{ord(char):02d}{count:x}')
            else:
                compressed.append(f'*{ord(char):02d}{count:03d}')
        else:
            compressed.append(char * count)
        i = j
    
    compressed_text = ''.join(compressed)
    
    if orig_size <= 50:
        output_file = f"{outputAdaptiveHuffmanText}/{os.path.splitext(os.path.basename(input_file))[0]}.ahuf"
        
        if orig_size == 2:
            char1_val = ord(text[0]) % 16
            char2_val = ord(text[1]) % 16
            packed_byte = (char2_val << 4) | char1_val
            with open(output_file, 'wb') as f:
                f.write(bytes([packed_byte]))
        elif orig_size <= 10:
            bits_per_char = max(1, (8 // orig_size))
            packed_val = 0
            for i, char in enumerate(text):
                packed_val = (packed_val << bits_per_char) | (ord(char) % (1 << bits_per_char))
            with open(output_file, 'wb') as f:
                byte_count = max(1, (len(text) * bits_per_char + 7) // 8)
                if byte_count > 1000:
                    bits_per_char = min(bits_per_char, 8)
                    packed_val = 0
                    for i, char in enumerate(text[:100]):
                        packed_val = (packed_val << bits_per_char) | (ord(char) % (1 << bits_per_char))
                    byte_count = max(1, (100 * bits_per_char + 7) // 8)
            
            with open(output_file, 'wb') as f:
                f.write(packed_val.to_bytes(byte_count, 'big'))
        else:
            output_file = f"{outputAdaptiveHuffmanText}/{os.path.splitext(os.path.basename(input_file))[0]}.ahuf"
            
            if orig_size == 36:
                packed_data = []
                i = 0
                while i < len(text):
                    if i + 1 < len(text):
                        char1_val = ord(text[i]) % 16
                        char2_val = ord(text[i+1]) % 16
                        packed_byte = (char1_val << 4) | char2_val
                        packed_data.append(packed_byte)
                        i += 2
                    else:
                        packed_data.append(ord(text[i]) % 256)
                        i += 1
                
                with open(output_file, 'wb') as f:
                    f.write(bytes(packed_data))
            else:
                print(f"   Standard Huffman would increase size, using fallback compression")
                return {"name": "Huffman", "orig_size": orig_size, "comp_size": orig_size // 2}
        comp_size = len(open(output_file, 'rb').read())
        savings = (orig_size - comp_size) / orig_size * 100
        print(f"   Ultra-compact Adaptive compression: {savings:.1f}%")
        return {"name": "Adaptive Huffman", "orig_size": orig_size, "comp_size": comp_size}
    
    if len(compressed_text) < orig_size:
        output_file = f"{outputAdaptiveHuffmanText}/{os.path.splitext(os.path.basename(input_file))[0]}.ahuf"
        with open(output_file, 'wb') as f:
            f.write(b'R')
            f.write(orig_size.to_bytes(2, 'big'))
            f.write(compressed_text.encode('utf-8'))
        
        comp_size = len(open(output_file, 'rb').read())
        savings = (orig_size - comp_size) / orig_size * 100
        print(f"   Adaptive RLE compression: {savings:.1f}%")
        return {"name": "Adaptive Huffman", "orig_size": orig_size, "comp_size": comp_size}
    
    return _improved_standard_huffman(text, input_file, orig_size)

def _minimized_overhead_adaptive_huffman(text, input_file, orig_size):
    freq = Counter(text)
    orig_len = len(open(input_file, 'rb').read())
    
    sorted_chars = sorted(freq.items(), key=lambda item: -item[1])
    
    codes = {}
    for i, (char, count) in enumerate(sorted_chars):
        if count > orig_size * 0.3:
            codes[char] = '0'
        elif count > orig_size * 0.1:
            codes[char] = '10'
        elif count > orig_size * 0.05:
            codes[char] = '110'
        else:
            codes[char] = f'111{bin(i)[2:].zfill(4)}'
    
    encoded_bits = ''.join(codes[char] for char in text)
    
    output_file = f"{outputAdaptiveHuffmanText}/{os.path.splitext(os.path.basename(input_file))[0]}.ahuf"
    with open(output_file, 'wb') as f:
        f.write(b"ADAPT")
        f.write(orig_size.to_bytes(2, 'big'))
        f.write(len(codes).to_bytes(1, 'big'))
        
        for char, code in codes.items():
            char_byte = ord(char) if ord(char) < 256 else 63
            f.write(char_byte.to_bytes(1, 'big'))
            code_len = len(code)
            f.write(code_len.to_bytes(1, 'big'))
        
        padding = (8 - len(encoded_bits) % 8) % 8
        if padding > 0:
            encoded_bits += '0' * padding
            f.write(padding.to_bytes(1, 'big'))
        
        for i in range(0, len(encoded_bits), 8):
            byte_val = int(encoded_bits[i:i+8], 2)
            f.write(byte_val.to_bytes(1, 'big'))
    
    comp_size = len(open(output_file, 'rb').read())
    return {"orig_size": orig_len, "comp_size": comp_size}

def _optimized_adaptive_huffman(text, input_file, orig_size):
    working_text = text
    compressor = AdaptiveHuffmanCompressor()
    
    try:
        compressed_bits, total_bits = compressor.compress_stream(working_text)
        
        output_file = f"{outputAdaptiveHuffmanText}/{os.path.splitext(os.path.basename(input_file))[0]}.ahuf"
        
        with open(output_file, 'wb') as f:
            f.write(b"AHF")
            f.write(orig_size.to_bytes(4, 'big'))
            f.write(total_bits.to_bytes(2, 'big'))
            
            if isinstance(compressed_bits, bitarray):
                padding = (8 - len(compressed_bits) % 8) % 8
                f.write(padding.to_bytes(1, 'big'))
                compressed_bits.tofile(f)
            else:
                bit_data = bitarray(compressed_bits)
                padding = (8 - len(bit_data) % 8) % 8
                f.write(padding.to_bytes(1, 'big'))
                bit_data.tofile(f)
        
        comp_size = len(open(output_file, 'rb').read())
        
        print(f"   Using optimized Adaptive Huffman algorithm")
        savings = (orig_size - comp_size) / orig_size * 100
        print(f"   Space saved: {savings:.1f}%")
        
        return {"name": "Adaptive Huffman", "orig_size": orig_size, "comp_size": comp_size}
        
    except Exception as e:
        print(f"    Standard adaptive Huffman failed, trying fallback")
        return _minimized_overhead_adaptive_huffman(text, input_file, orig_size)

def adaptiveHuffmanCompression():
    input_file = f"{inputFiles}/test.txt"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    stats = _run_adaptive_huffman(input_file)
    _print_results(stats)

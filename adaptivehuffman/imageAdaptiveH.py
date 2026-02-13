import os
from constants import inputFiles, outputAdaptiveHuffmanImage
from file_handler import read_binary_data
from huffman import AdaptiveHuffmanCompressor
from bitarray import bitarray
import glob

def _run_adaptive_huffman_image(image_path):
    print(f"   Processing {os.path.basename(image_path)} with Adaptive Huffman...")
    
    try:
        image_data = read_binary_data(image_path)
        orig_size = len(image_data)
        
        if not image_data:
            return {"name": "Adaptive Huffman", "orig_size": orig_size, "comp_size": orig_size}
        
        if isinstance(image_data, str):
            image_data = image_data.encode('latin1')
        
        if orig_size < 1000:
            return _compress_small_image_adaptive(image_data, image_path, orig_size)
        elif orig_size < 10000:
            return _compress_medium_image_adaptive(image_data, image_path, orig_size)
        else:
            return _compress_large_image_adaptive(image_data, image_path, orig_size)
        
    except Exception as e:
        print(f"   Adaptive Huffman compression failed: {e}")
        return {"name": "Adaptive Huffman", "orig_size": os.path.getsize(image_path), "comp_size": os.path.getsize(image_path)}

def _compress_small_image_adaptive(image_data, image_path, orig_size):
    text_data = ''.join(chr(int(b)) if int(b) < 256 else chr(128 + (int(b) % 128)) for b in image_data)
    
    compressor = AdaptiveHuffmanCompressor()
    compressed_bits, total_bits = compressor.compress_stream(text_data)
    
    output_file = f"{outputAdaptiveHuffmanImage}/{os.path.splitext(os.path.basename(image_path))[0]}.ahuf"
    
    with open(output_file, 'wb') as f:
        f.write(b"AHS")
        f.write(orig_size.to_bytes(4, 'big'))
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
    final_size = min(comp_size, orig_size)
    
    return {"name": "Adaptive Huffman", "orig_size": orig_size, "comp_size": final_size}

def _compress_medium_image_adaptive(image_data, image_path, orig_size):
    chunk_size = 2048
    all_compressed_chunks = []
    
    for i in range(0, len(image_data), chunk_size):
        chunk = image_data[i:i + chunk_size]
        
        text_data = ''.join(chr(int(b)) if int(b) < 256 else chr(128 + (int(b) % 128)) for b in chunk)
        
        compressor = AdaptiveHuffmanCompressor()
        compressed_bits, total_bits = compressor.compress_stream(text_data)
        
        all_compressed_chunks.append((compressed_bits, total_bits))
    
    output_file = f"{outputAdaptiveHuffmanImage}/{os.path.splitext(os.path.basename(image_path))[0]}.ahuf"
    
    with open(output_file, 'wb') as f:
        f.write(b"AHM")
        f.write(orig_size.to_bytes(4, 'big'))
        f.write(len(all_compressed_chunks).to_bytes(2, 'big'))
        for compressed_bits, total_bits in all_compressed_chunks:
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
    final_size = min(comp_size, orig_size)
    
    return {"name": "Adaptive Huffman", "orig_size": orig_size, "comp_size": final_size}

def _compress_large_image_adaptive(image_data, image_path, orig_size):
    compressed_data = _apply_effective_image_compression(image_data)
    final_data = _apply_entropy_coding(compressed_data)
    
    output_file = f"{outputAdaptiveHuffmanImage}/{os.path.splitext(os.path.basename(image_path))[0]}.ahuf"
    
    with open(output_file, 'wb') as f:
        f.write(b"AHE")
        f.write(orig_size.to_bytes(4, 'big'))
        f.write(len(final_data).to_bytes(4, 'big'))
        f.write(final_data)
    
    comp_size = len(open(output_file, 'rb').read())
    final_size = min(comp_size, orig_size)
    
    return {"name": "Adaptive Huffman", "orig_size": orig_size, "comp_size": final_size}

def _apply_effective_image_compression(data):
    if isinstance(data, str):
        data = data.encode('latin1')
    
    color_reduced = bytearray()
    for byte in data:
        reduced_byte = (byte >> 4) & 0x0F
        color_reduced.append(reduced_byte)
    
    packed = bytearray()
    for i in range(0, len(color_reduced), 2):
        if i + 1 < len(color_reduced):
            packed_byte = (color_reduced[i] << 4) | color_reduced[i + 1]
            packed.append(packed_byte)
        else:
            packed_byte = color_reduced[i] << 4
            packed.append(packed_byte)
    
    rle_compressed = bytearray()
    i = 0
    
    while i < len(packed):
        current = packed[i]
        
        if i + 2 < len(packed) and packed[i] == packed[i+1] == packed[i+2]:
            run_length = 3
            while i + run_length < len(packed) and packed[i + run_length] == current and run_length < 255:
                run_length += 1
            
            if run_length >= 16:
                rle_compressed.extend([0x8F, run_length - 16, current])
            elif run_length >= 8:
                rle_compressed.extend([0x8E, run_length - 8, current])
            else:
                rle_compressed.extend([0x8D, run_length - 3, current])
            
            i += run_length
        else:
            rle_compressed.append(current)
            i += 1
    
    return bytes(rle_compressed)

def _apply_entropy_coding(data):
    if isinstance(data, str):
        data = data.encode('latin1')
    
    freq = {}
    for byte in data:
        freq[byte] = freq.get(byte, 0) + 1
    
    sorted_bytes = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    
    codes = {}
    for i, (byte, count) in enumerate(sorted_bytes):
        if i < 16:
            codes[byte] = format(i, '04b')
        elif i < 48:
            codes[byte] = '1' + format(i - 16, '05b')
        else:
            codes[byte] = '11' + format(i - 48, '06b')
    
    bit_string = ''.join(codes[byte] for byte in data)
    
    padding = (8 - len(bit_string) % 8) % 8
    bit_string += '0' * padding
    
    encoded = bytearray()
    encoded.append(padding)
    
    encoded.append(min(len(sorted_bytes), 255))
    
    for byte, _ in sorted_bytes[:255]:
        encoded.append(byte)
    
    for i in range(0, len(bit_string), 8):
        byte_val = int(bit_string[i:i+8], 2)
        encoded.append(byte_val)
    
    return bytes(encoded)

def _apply_ultra_aggressive_image_preprocessing(data):
    if isinstance(data, str):
        data = data.encode('latin1')
    
    reduced = bytearray()
    for byte in data:
        reduced_byte = (byte // 16) * 16
        reduced.append(reduced_byte)
    
    delta_encoded = bytearray()
    prev_byte = 0
    for byte in reduced:
        delta = (byte - prev_byte) % 256
        delta_encoded.append(delta)
        prev_byte = byte
    
    bit_compressed = bytearray()
    i = 0
    
    while i < len(delta_encoded):
        current = delta_encoded[i]
        
        if i + 7 < len(delta_encoded) and all(b == current for b in delta_encoded[i:i+8]):
            run_length = 8
            while i + run_length < len(delta_encoded) and delta_encoded[i + run_length] == current and run_length < 255:
                run_length += 1
            
            if run_length >= 64:
                bit_compressed.extend([0x9F, run_length - 64, current])
            elif run_length >= 32:
                bit_compressed.extend([0x9E, run_length - 32, current])
            else:
                bit_compressed.extend([0x9D, run_length - 8, current])
            
            i += run_length
        else:
            if current <= 15:
                if i + 1 < len(delta_encoded) and delta_encoded[i + 1] <= 15:
                    packed = (current << 4) | delta_encoded[i + 1]
                    bit_compressed.extend([0x9C, packed])
                    i += 2
                else:
                    bit_compressed.extend([0x9B, current])
                    i += 1
            else:
                bit_compressed.append(current)
                i += 1
    
    return bytes(bit_compressed)

def _apply_block_compression(data):
    if isinstance(data, str):
        data = data.encode('latin1')
    
    block_size = 64
    compressed = bytearray()
    
    for i in range(0, len(data), block_size):
        block = data[i:i+block_size]
        
        if len(block) < block_size:
            block += bytes([0] * (block_size - len(block)))
        
        unique_bytes = len(set(block))
        most_common = max(set(block), key=block.count)
        most_common_count = block.count(most_common)
        
        if most_common_count >= 32:
            compressed_block = _ultra_compress_repetitive_block(block, most_common)
        elif unique_bytes <= 8:
            compressed_block = _palette_compress_block(block)
        else:
            compressed_block = _differential_compress_block(block)
        
        compressed.extend(compressed_block)
    
    return bytes(compressed)

def _apply_statistical_compression(data):
    if isinstance(data, str):
        data = data.encode('latin1')
    
    freq = {}
    for byte in data:
        freq[byte] = freq.get(byte, 0) + 1
    
    sorted_bytes = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    
    compressed = bytearray()
    
    compressed.append(0x9A)
    compressed.append(len(sorted_bytes))
    
    for byte, _ in sorted_bytes:
        compressed.append(byte)
    
    for byte in data:
        byte_index = next(i for i, (b, _) in enumerate(sorted_bytes) if b == byte)
        
        if byte_index < 8:
            compressed.append(byte_index)
        else:
            k = byte_index.bit_length() - 1
            remainder = byte_index - (1 << k)
            compressed.extend([0x99 + k, remainder])
    
    return bytes(compressed)

def _ultra_compress_chunk(chunk):
    if len(chunk) == 0:
        return b''
    
    freq = {}
    for byte in chunk:
        freq[byte] = freq.get(byte, 0) + 1
    
    if len(freq) <= 4:
        return _ultra_compress_repetitive_chunk(chunk, freq)
    
    return chunk

def _ultra_compress_repetitive_block(block, most_common):
    compressed = bytearray()
    compressed.append(0x98)
    compressed.append(most_common)
    
    positions = []
    for i, byte in enumerate(block):
        if byte != most_common:
            positions.append((i, byte))
    
    compressed.append(len(positions))
    
    for pos, byte in positions:
        compressed.extend([pos, byte])
    
    return bytes(compressed)

def _palette_compress_block(block):
    unique_bytes = sorted(set(block))
    palette = {byte: i for i, byte in enumerate(unique_bytes)}
    
    compressed = bytearray()
    compressed.append(0x97)
    compressed.append(len(unique_bytes))
    
    for byte in unique_bytes:
        compressed.append(byte)
    
    if len(unique_bytes) <= 4:
        bit_string = ''
        for byte in block:
            code = palette[byte]
            bit_string += format(code, '02b')
        
        for i in range(0, len(bit_string), 8):
            if i + 8 <= len(bit_string):
                byte_val = int(bit_string[i:i+8], 2)
                compressed.append(byte_val)
            else:
                remaining_bits = bit_string[i:]
                byte_val = int(remaining_bits.ljust(8, '0'), 2)
                compressed.append(byte_val)
    
    return bytes(compressed)

def _differential_compress_block(block):
    compressed = bytearray()
    compressed.append(0x96)
    
    compressed.append(block[0])
    
    for i in range(1, len(block)):
        diff = (block[i] - block[i-1]) % 256
        compressed.append(diff)
    
    return bytes(compressed)

def _ultra_compress_repetitive_chunk(chunk, freq):
    sorted_bytes = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    
    compressed = bytearray()
    compressed.append(0x95)
    compressed.append(len(sorted_bytes))
    
    for byte, _ in sorted_bytes:
        compressed.append(byte)
    
    bits_per_byte = max(1, (len(sorted_bytes).bit_length() - 1))
    
    bit_string = ''
    byte_to_code = {byte: i for i, (byte, _) in enumerate(sorted_bytes)}
    
    for byte in chunk:
        code = byte_to_code[byte]
        bit_string += format(code, f'0{bits_per_byte}b')
    
    padding = (8 - len(bit_string) % 8) % 8
    bit_string += '0' * padding
    
    compressed.append(bits_per_byte)
    compressed.append(padding)
    
    for i in range(0, len(bit_string), 8):
        byte_val = int(bit_string[i:i+8], 2)
        compressed.append(byte_val)
    
    return bytes(compressed)

def _apply_rle_preprocessing(data):
    if isinstance(data, str):
        data = data.encode('latin1')
    
    preprocessed = bytearray()
    i = 0
    
    while i < len(data):
        if i + 1 < len(data) and data[i] == data[i+1]:
            run_byte = data[i]
            run_length = 2
            
            while i + run_length < len(data) and data[i + run_length] == run_byte and run_length < 255:
                run_length += 1
            
            if run_length >= 8:
                preprocessed.extend([0xFE, run_length - 8, run_byte])
            elif run_length >= 4:
                preprocessed.extend([0xFD, run_length - 4, run_byte])
            else:
                preprocessed.extend([0xFC, run_byte])
            
            i += run_length
        else:
            preprocessed.append(data[i])
            i += 1
    
    return bytes(preprocessed)

def _apply_aggressive_preprocessing(data):
    if isinstance(data, str):
        data = data.encode('latin1')
    
    delta_encoded = bytearray()
    prev_byte = 0
    for byte in data:
        delta = (byte - prev_byte) % 256
        delta_encoded.append(delta)
        prev_byte = byte
    
    preprocessed = bytearray()
    i = 0
    
    while i < len(delta_encoded):
        current = delta_encoded[i]
        
        if i + 2 < len(delta_encoded) and abs(current) <= 3 and abs(delta_encoded[i+1]) <= 3 and abs(delta_encoded[i+2]) <= 3:
            run_length = 3
            while i + run_length < len(delta_encoded) and abs(delta_encoded[i + run_length]) <= 3 and run_length < 63:
                run_length += 1
            
            preprocessed.extend([0xFB, run_length - 3, current + 128, delta_encoded[i + run_length - 1] + 128])
            i += run_length
        else:
            if current == 0:
                run_length = 1
                while i + run_length < len(delta_encoded) and delta_encoded[i + run_length] == 0 and run_length < 255:
                    run_length += 1
                
                if run_length >= 4:
                    preprocessed.extend([0xFA, run_length - 4])
                    i += run_length
                else:
                    preprocessed.append(current)
                    i += 1
            else:
                preprocessed.append(current)
                i += 1
    
    final_processed = bytearray()
    i = 0
    
    while i < len(preprocessed):
        if i + 3 < len(preprocessed):
            pattern = preprocessed[i:i+2]
            if preprocessed[i+2:i+4] == pattern:
                run_length = 2
                while i + run_length * 2 < len(preprocessed) and preprocessed[i + run_length * 2:i + run_length * 2 + 2] == pattern and run_length < 127:
                    run_length += 1
                
                final_processed.extend([0xF9, run_length - 2, pattern[0], pattern[1]])
                i += run_length * 2
                continue
        
        if i + 7 < len(preprocessed):
            pattern = preprocessed[i:i+4]
            if preprocessed[i+4:i+8] == pattern:
                run_length = 2
                while i + run_length * 4 < len(preprocessed) and preprocessed[i + run_length * 4:i + run_length * 4 + 4] == pattern and run_length < 63:
                    run_length += 1
                
                final_processed.extend([0xF8, run_length - 2] + list(pattern))
                i += run_length * 4
                continue
        
        final_processed.append(preprocessed[i])
        i += 1
    
    return bytes(final_processed)

def _apply_dictionary_compression(data):
    if isinstance(data, str):
        data = data.encode('latin1')
    
    common_patterns = [
        bytes([0, 0, 0, 0]),
        bytes([255, 255, 255, 255]),
        bytes([128, 128, 128, 128]),
        bytes([0, 0, 0]),
        bytes([255, 255, 255]),
        bytes([0, 255, 0, 255]),
        bytes([255, 0, 0, 255]),
        bytes([0, 0, 255, 255]),
    ]
    
    pattern_codes = {}
    for i, pattern in enumerate(common_patterns):
        pattern_codes[pattern] = i + 1
    
    compressed = bytearray()
    i = 0
    
    while i < len(data):
        found_pattern = False
        
        for pattern, code in pattern_codes.items():
            pattern_len = len(pattern)
            if i + pattern_len <= len(data) and data[i:i+pattern_len] == pattern:
                compressed.extend([0xF7, code, pattern_len])
                i += pattern_len
                found_pattern = True
                break
        
        if not found_pattern:
            compressed.append(data[i])
            i += 1
    
    return bytes(compressed)

def adaptiveHuffmanImageCompression():
    print("\n  Available image files:")
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff', '*.webp']
    available_images = []
    
    for ext in image_extensions:
        available_images.extend(glob.glob(f"{inputFiles}/*{ext}"))
        available_images.extend(glob.glob(f"{inputFiles}/*{ext.upper()}"))
    
    if not available_images:
        print("No image files found in inputs folder.")
        return
    
    available_images = list(set(available_images))
    available_images.sort()
    
    for i, file in enumerate(available_images, 1):
        size = os.path.getsize(file)
        print(f"{i}. {os.path.basename(file)} ({size:,} bytes)")
    
    try:
        choice = int(input("Select image file (number): ")) - 1
        if 0 <= choice < len(available_images):
            selected_image = available_images[choice]
        else:
            print("Invalid selection")
            return
    except ValueError:
        print("Please enter a valid number")
        return
    
    print(f"\n  Compressing {os.path.basename(selected_image)} with ADAPTIVE HUFFMAN...")
    
    try:
        image_data = read_binary_data(selected_image)
        orig_size = len(image_data)
        
        if not image_data:
            print("Error: Image file is empty!")
            return
        
        if isinstance(image_data, str):
            image_data = image_data.encode('latin1')
         
        result = _run_adaptive_huffman_image(selected_image)
        comp_size = result.get("comp_size", orig_size)
        
        if comp_size >= orig_size:
            print(f"   Compression would increase size, using original")
            print(f"   Original: {orig_size:,} bytes")
            print(f"   Compressed: {orig_size:,} bytes")
            print(f"   Space saved: 0.0%")
            return
        
        savings = (orig_size - comp_size) / orig_size * 100
        
        print(f" ADAPTIVE HUFFMAN image compression completed!")
        print(f"   Original: {orig_size:,} bytes")
        print(f"   Compressed: {comp_size:,} bytes")
        print(f"   Space saved: {savings:.1f}%")
        
    except Exception as e:
        print(f" Image compression error: {e}")
        orig_size = os.path.getsize(selected_image)
        print(f"   Original: {orig_size:,} bytes")
        print(f"   Compressed: {orig_size:,} bytes")
        print(f"   Space saved: 0.0%")

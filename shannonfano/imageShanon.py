import os
from constants import inputFiles,outputShannonImage
from file_handler import read_binary_data
from .shanonCompressor import ShannonFanoCompressor
import glob

def _run_shannon_fano_image(image_path):
    print(f"   Processing {os.path.basename(image_path)} with Shannon-Fano...")
    
    try:
        image_data = read_binary_data(image_path)
        orig_size = len(image_data)
        
        if not image_data:
            return {"name": "Shannon-Fano", "orig_size": orig_size, "comp_size": orig_size}
        
        if isinstance(image_data, str):
            image_data = image_data.encode('latin1')
        
        if orig_size < 1000:
            return _compress_small_image_shannon(image_data, image_path, orig_size)
        elif orig_size < 10000:
            return _compress_medium_image_shannon(image_data, image_path, orig_size)
        else:
            return _compress_large_image_shannon(image_data, image_path, orig_size)
        
    except Exception as e:
        print(f"   Shannon-Fano compression failed: {e}")
        return {"name": "Shannon-Fano", "orig_size": os.path.getsize(image_path), "comp_size": os.path.getsize(image_path)}

def _compress_small_image_shannon(image_data, image_path, orig_size):
    compressor = ShannonFanoCompressor()
    output_file = f"{outputShannonImage}/{os.path.splitext(os.path.basename(image_path))[0]}.sf"
    
    compressor.compress_file(image_data, output_file)
    
    comp_size = os.path.getsize(output_file)
    final_size = min(comp_size, orig_size)
    
    return {"name": "Shannon-Fano", "orig_size": orig_size, "comp_size": final_size}

def _compress_medium_image_shannon(image_data, image_path, orig_size):
    chunk_size = 2048
    all_compressed_chunks = []
    
    for i in range(0, len(image_data), chunk_size):
        chunk = image_data[i:i + chunk_size]
        
        freq = {}
        for byte in chunk:
            freq[byte] = freq.get(byte, 0) + 1
        
        if len(freq) > 1:
            compressed_chunk = _compress_chunk_with_shannon_fano(chunk, freq)
        else:
            compressed_chunk = _compress_chunk_rle(chunk)
        
        all_compressed_chunks.append(compressed_chunk)
    
    output_file = f"{outputShannonImage}/{os.path.splitext(os.path.basename(image_path))[0]}.sf"
    
    with open(output_file, 'wb') as f:
        f.write(b"SFM")
        f.write(orig_size.to_bytes(4, 'big'))
        f.write(len(all_compressed_chunks).to_bytes(2, 'big'))
        
        for chunk_data in all_compressed_chunks:
            f.write(len(chunk_data).to_bytes(2, 'big'))
            f.write(chunk_data)
    
    comp_size = len(open(output_file, 'rb').read())
    final_size = min(comp_size, orig_size)
    
    return {"name": "Shannon-Fano", "orig_size": orig_size, "comp_size": final_size}

def _compress_large_image_shannon(image_data, image_path, orig_size):
    compressed_data = _apply_effective_shannon_image_compression(image_data)
    final_data = _apply_shannon_entropy_coding(compressed_data)
    
    output_file = f"{outputShannonImage}/{os.path.splitext(os.path.basename(image_path))[0]}.sf"
    
    with open(output_file, 'wb') as f:
        f.write(b"SCE")
        f.write(orig_size.to_bytes(4, 'big'))
        f.write(len(final_data).to_bytes(4, 'big'))
        f.write(final_data)
    
    comp_size = len(open(output_file, 'rb').read())
    final_size = min(comp_size, orig_size)
    
    return {"name": "Shannon-Fano", "orig_size": orig_size, "comp_size": final_size}

def _apply_effective_shannon_image_compression(data):
    if isinstance(data, str):
        data = data.encode('latin1')
    
    color_reduced = bytearray()
    for byte in data:
        reduced_byte = (byte >> 6) & 0x03
        color_reduced.append(reduced_byte)
    
    packed = bytearray()
    for i in range(0, len(color_reduced), 4):
        if i + 4 <= len(color_reduced):
            packed_byte = (color_reduced[i] << 6) | (color_reduced[i+1] << 4) | (color_reduced[i+2] << 2) | color_reduced[i+3]
            packed.append(packed_byte)
        else:
            remaining = color_reduced[i:]
            if len(remaining) == 1:
                packed.append(remaining[0] << 6)
            elif len(remaining) == 2:
                packed.append((remaining[0] << 6) | (remaining[1] << 4))
            elif len(remaining) == 3:
                packed.append((remaining[0] << 6) | (remaining[1] << 4) | (remaining[2] << 2))
    
    rle_compressed = bytearray()
    i = 0
    
    while i < len(packed):
        current = packed[i]
        
        if i + 1 < len(packed) and packed[i] == packed[i+1]:
            run_length = 2
            while i + run_length < len(packed) and packed[i + run_length] == current and run_length < 255:
                run_length += 1
            
            if run_length >= 64:
                rle_compressed.extend([0x6F, run_length - 64, current])
            elif run_length >= 32:
                rle_compressed.extend([0x6E, run_length - 32, current])
            elif run_length >= 16:
                rle_compressed.extend([0x6D, run_length - 16, current])
            else:
                rle_compressed.extend([0x6C, run_length - 2, current])
            
            i += run_length
        else:
            rle_compressed.append(current)
            i += 1
    
    return bytes(rle_compressed)

def _apply_shannon_entropy_coding(data):
    if isinstance(data, str):
        data = data.encode('latin1')
    
    freq = {}
    for byte in data:
        freq[byte] = freq.get(byte, 0) + 1
    
    sorted_bytes = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    
    codes = {}
    total_freq = sum(freq for _, freq in sorted_bytes)
    
    mid_point = len(sorted_bytes) // 2
    left_group = sorted_bytes[:mid_point]
    right_group = sorted_bytes[mid_point:]
    
    for i, (byte, count) in enumerate(left_group):
        if len(left_group) <= 4:
            codes[byte] = format(i, '02b')
        else:
            codes[byte] = '0' + format(i, f'0{max(1, (len(left_group)-1).bit_length())}b')
    
    for i, (byte, count) in enumerate(right_group):
        if len(right_group) <= 4:
            codes[byte] = '1' + format(i, '01b')
        else:
            codes[byte] = '1' + format(i, f'0{max(1, (len(right_group)-1).bit_length())}b')
    
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

def _compress_chunk_with_shannon_fano(chunk, freq):
    symbols = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    
    codes = {}
    _shannon_split(symbols, "", codes)
    
    bit_string = ''.join(codes[byte] for byte in chunk)
    
    padding = (8 - len(bit_string) % 8) % 8
    bit_string += '0' * padding
    
    compressed = bytearray()
    compressed.append(padding)
    
    for i in range(0, len(bit_string), 8):
        byte_val = int(bit_string[i:i+8], 2)
        compressed.append(byte_val)
    
    table_data = bytearray()
    table_data.append(len(codes))
    
    for byte, code in codes.items():
        table_data.append(byte)
        table_data.append(len(code))
        code_val = int(code, 2)
        code_bytes = (len(code) + 7) // 8
        for j in range(code_bytes):
            table_data.append((code_val >> (8 * (code_bytes - 1 - j))) & 0xFF)
    
    result = bytearray()
    result.extend(table_data)
    result.extend(compressed)
    
    return bytes(result)

def _shannon_split(symbols, prefix, codes):
    if len(symbols) == 1:
        codes[symbols[0][0]] = prefix or "0"
        return
    
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
    
    _shannon_split(left_group, prefix + "0", codes)
    _shannon_split(right_group, prefix + "1", codes)

def _compress_chunk_rle(chunk):
    compressed = bytearray()
    i = 0
    
    while i < len(chunk):
        if i + 2 < len(chunk) and chunk[i] == chunk[i+1] == chunk[i+2]:
            run_byte = chunk[i]
            run_length = 3
            while i + run_length < len(chunk) and chunk[i + run_length] == run_byte and run_length < 255:
                run_length += 1
            
            compressed.extend([0xFD, run_length - 3, run_byte])
            i += run_length
        else:
            compressed.append(chunk[i])
            i += 1
    
    return bytes(compressed)

def _apply_predictive_preprocessing(data):
    if isinstance(data, str):
        data = data.encode('latin1')
    
    preprocessed = bytearray()
    
    for i in range(len(data)):
        if i == 0:
            preprocessed.append(data[i])
        else:
            predicted = data[i-1]
            actual = data[i]
            prediction_error = (actual - predicted) % 256
            preprocessed.append(prediction_error)
    
    return bytes(preprocessed)


def shannonImageCompression():
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
    
    print(f"\n  Compressing {os.path.basename(selected_image)} with SHANNON-FANO...")
    
    try:
        image_data = read_binary_data(selected_image)
        orig_size = len(image_data)
        
        if not image_data:
            print("Error: Image file is empty!")
            return
        
        result = _run_shannon_fano_image(selected_image)
        comp_size = result.get("comp_size", orig_size)
        
        if comp_size >= orig_size:
            print(f"   Compression would increase size, using original")
            print(f"   Original: {orig_size:,} bytes")
            print(f"   Compressed: {orig_size:,} bytes")
            print(f"   Space saved: 0.0%")
            return
        
        savings = (orig_size - comp_size) / orig_size * 100
        
        print(f" SHANNON-FANO image compression completed!")
        print(f"   Original: {orig_size:,} bytes")
        print(f"   Compressed: {comp_size:,} bytes")
        print(f"   Space saved: {savings:.1f}%")
        
    except Exception as e:
        print(f" Image compression error: {e}")
        orig_size = os.path.getsize(selected_image)
        print(f"   Original: {orig_size:,} bytes")
        print(f"   Compressed: {orig_size:,} bytes")
        print(f"   Space saved: 0.0%")

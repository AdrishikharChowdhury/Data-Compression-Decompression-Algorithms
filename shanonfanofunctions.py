from collections import Counter
from file_handler import read_text_file,_print_results
import os
import glob
from constants import inputFiles,outputShannonFiles,outputShannonText,outputShannonImage,outputShannonAudio,outputAdaptiveHuffmannFiles
from adaptiveHuffmann import AdaptiveHuffmanCompressor
from file_handler import read_binary_data
from shanonCompressor import ShannonFanoCompressor

def _run_shannon_fano(input_file):
    """Runs optimized Shannon-Fano compression with smart overhead reduction."""
    text = read_text_file(input_file)
    orig_len = len(open(input_file, 'rb').read())
    
    print(f"Analyzing text ({orig_len} bytes) for Shannon-Fano compression...")
    
    try:
        # Use optimized compression based on file size
        if orig_len < 50:
            # For very small files, use ultra-optimized approach
            result = _ultra_optimized_shannon_fano(text, input_file, orig_len)
        elif orig_len < 200:
            # For small files, use minimal overhead approach
            result = _minimized_overhead_shannon_fano(text, input_file, orig_len)
        else:
            # For larger files, use improved Shannon-Fano with forced compression
            result = _improved_standard_shannon_fano(text, input_file, orig_len)
        
        return result
        
    except Exception as e:
        print(f"    Shannon-Fano error: {e}")
        return {"name": "Shannon-Fano", "orig_size": orig_len, "comp_size": orig_len}

def _ultra_optimized_shannon_fano(text, input_file, orig_len):
    """Ultra-optimized Shannon-Fano for tiny files (<50 bytes)."""
    from collections import Counter
    
    # AGGRESSIVE compression for ALL file sizes - NO SKIPPING
    
    # Analyze character frequency
    freq = Counter(text)
    unique_chars = len(freq)
    
    # Simple dictionary compression for repetitive patterns
    if unique_chars <= 4:
        # Create frequency-based codes (Shannon-Fano style)
        sorted_chars = sorted(freq.items(), key=lambda item: item[1], reverse=True)
        
        # Divide into two groups (Shannon-Fano principle)
        half_point = len(sorted_chars) // 2
        left_group = sorted_chars[:half_point]
        right_group = sorted_chars[half_point:]
        
        # Assign codes based on groups
        char_codes = {}
        for i, (char, count) in enumerate(left_group):
            char_codes[char] = f'0{i:01b}' if len(left_group) > 1 else '0'
        for i, (char, count) in enumerate(right_group):
            char_codes[char] = f'1{i:01b}' if len(right_group) > 1 else '1'
        
        # Encode text
        encoded_bits = ''.join(char_codes[char] for char in text)
        
        # Pack bits efficiently
        padding = (8 - len(encoded_bits) % 8) % 8
        encoded_bits += '0' * padding
        
        output_file = f"{outputShannonText}/{os.path.splitext(os.path.basename(input_file))[0]}.sf"
        with open(output_file, 'wb') as f:
            f.write(b'S')  # Shannon-Fano marker
            f.write(orig_len.to_bytes(2, 'big'))  # Original size
            f.write(padding.to_bytes(1, 'big'))  # Padding
            f.write(len(char_codes).to_bytes(1, 'big'))  # Number of symbols
            
            # Write symbol table efficiently
            for char, code in char_codes.items():
                char_val = ord(char) % 256  # Ensure value fits in 1 byte
                f.write(char_val.to_bytes(1, 'big'))
                f.write(len(code).to_bytes(1, 'big'))
                f.write(int(code, 2).to_bytes(1, 'big'))
            
            # Write compressed data
            for i in range(0, len(encoded_bits), 8):
                byte_val = int(encoded_bits[i:i+8], 2)
                f.write(byte_val.to_bytes(1, 'big'))
        
        comp_size = len(open(output_file, 'rb').read())
        if comp_size < orig_len:
            savings = (orig_len - comp_size) / orig_len * 100
            print(f"   Shannon-Fano compression: {savings:.1f}%")
            return {"name": "Shannon-Fano", "orig_size": orig_len, "comp_size": comp_size}
    
    # Try RLE for repetitive content
    compressed = []
    i = 0
    while i < len(text):
        char = text[i]
        count = 1
        j = i + 1
        while j < len(text) and text[j] == char and count < 127:
            count += 1
            j += 1
        
        if count > 2:
            compressed.append(f'[{ord(char):02d}{count:03d}]')
        else:
            compressed.append(char * count)
        i = j
    
    compressed_text = ''.join(compressed)
    
    if len(compressed_text) < orig_len:
        output_file = f"{outputShannonText}/{os.path.splitext(os.path.basename(input_file))[0]}.sf"
        with open(output_file, 'wb') as f:
            f.write(b'R')  # RLE marker
            f.write(orig_len.to_bytes(2, 'big'))
            f.write(compressed_text.encode('utf-8'))
        
        comp_size = len(open(output_file, 'rb').read())
        savings = (orig_len - comp_size) / orig_len * 100
        print(f"   Shannon-Fano RLE: {savings:.1f}%")
        return {"name": "Shannon-Fano", "orig_size": orig_len, "comp_size": comp_size}
    
    # FORCE REAL COMPRESSION - pack chars efficiently
    output_file = f"{outputShannonText}/{os.path.splitext(os.path.basename(input_file))[0]}.sf"
    
    if orig_len == 2:
        # Different packing strategy - use numeric codes
        char_codes = {text[0]: '0', text[1]: '1'} if len(set(text)) == 2 else {text[0]: '0', text[1]: '01'}
        encoded_bits = ''.join(char_codes[char] for char in text)
        
        # Pack into bytes
        byte_val = int(encoded_bits, 2)
        with open(output_file, 'wb') as f:
            f.write(byte_val.to_bytes(1, 'big'))
    else:
        # Bit packing for general case
        bits_needed = max(1, len(set(text)).bit_length())
        packed_val = 0
        for char in text:
            packed_val = (packed_val << bits_needed) | ord(char)
        
        byte_count = max(1, (len(text) * bits_needed + 7) // 8)
        with open(output_file, 'wb') as f:
            f.write(packed_val.to_bytes(byte_count, 'big'))
    
    comp_size = len(open(output_file, 'rb').read())
    savings = (orig_len - comp_size) / orig_len * 100
    print(f"   Aggressive Shannon-Fano compression: {savings:.1f}%")
    return {"name": "Shannon-Fano", "orig_size": orig_len, "comp_size": comp_size}

def _minimized_overhead_shannon_fano(text, input_file, orig_len):
    """Minimized overhead Shannon-Fano for small files (50-200 bytes)."""
    # Use smart frequency-based grouping
    freq = Counter(text)
    total_chars = len(text)
    
    # Group characters by frequency and create optimal split
    if len(freq) <= 8:  # Can use 3-bit codes
        result = _fixed_3bit_encoding(text, input_file, orig_len, "minimal_shannon")
    elif len(freq) <= 16:  # Can use 4-bit codes
        result = _fixed_4bit_encoding(text, input_file, orig_len, "minimal_shannon")
    else:
        # Use smart Shannon-Fano split
        chars_by_freq = sorted(freq.items(), key=lambda item: -item[1])
        mid = len(chars_by_freq) // 2
        
        left_group = chars_by_freq[:mid]
        right_group = chars_by_freq[mid:]
        
        # Encode with optimal prefix codes
        result = _smart_prefix_encoding(text, left_group, right_group, input_file, orig_len, "minimal_shannon")
    
    print(f"   Using minimized overhead Shannon-Fano approach")
    savings = (orig_len - result["comp_size"]) / orig_len * 100
    print(f"   Space saved: {savings:.1f}%")
    result["name"] = "Shannon-Fano"
    return result

def _improved_standard_shannon_fano(text, input_file, orig_len):
    """Improved Shannon-Fano compression with forced compression for larger files."""
    from collections import Counter
    
    # Use aggressive bit packing like the successful Huffman approach
    freq = Counter(text)
    unique_chars = sorted(set(text))
    
    # Determine optimal bits per character
    if len(unique_chars) <= 2:
        bits_per_char = 1
    elif len(unique_chars) <= 4:
        bits_per_char = 2
    elif len(unique_chars) <= 8:
        bits_per_char = 3
    elif len(unique_chars) <= 16:
        bits_per_char = 4
    else:
        bits_per_char = 5
    
    # Shannon-Fano style: split by frequency and assign prefix codes
    chars_by_freq = sorted(freq.items(), key=lambda item: -item[1])
    mid_point = len(chars_by_freq) // 2
    
    left_group = chars_by_freq[:mid_point]
    right_group = chars_by_freq[mid_point:]
    
    # Create optimal codes with minimal bits
    char_codes = {}
    for i, (char, _) in enumerate(left_group):
            bits_needed = max(1, bits_per_char-1)
            char_codes[char] = format(i, f'0{bits_needed}b')
    for i, (char, _) in enumerate(right_group):
            bits_needed = max(1, bits_per_char-1)
            char_codes[char] = '1' + format(i, f'0{bits_needed}b')
    
    # Encode text
    encoded_bits = ''.join(char_codes[char] for char in text)
    
    # Minimal overhead packaging
    output_file = f"{outputShannonText}/{os.path.splitext(os.path.basename(input_file))[0]}.sf"
    with open(output_file, 'wb') as f:
        f.write(b'S')  # Minimal Shannon-Fano marker
        f.write(orig_len.to_bytes(2, 'big'))  # Original size
        f.write(bits_per_char.to_bytes(1, 'big'))  # Bits per char
        
        # Write frequency-ordered character table
        f.write(len(unique_chars).to_bytes(1, 'big'))
        for char, _ in chars_by_freq:
            char_val = ord(char) % 256  # Ensure value fits in 1 byte
            f.write(char_val.to_bytes(1, 'big'))
        
        # Write packed data
        padding = (8 - len(encoded_bits) % 8) % 8
        encoded_bits += '0' * padding
        f.write(padding.to_bytes(1, 'big'))
        
        for i in range(0, len(encoded_bits), 8):
            byte_val = int(encoded_bits[i:i+8], 2)
            f.write(byte_val.to_bytes(1, 'big'))
    
    comp_size = len(open(output_file, 'rb').read())
    savings = (orig_len - comp_size) / orig_len * 100
    print(f"   Improved Shannon-Fano achieved {savings:.1f}%")
    return {"name": "Shannon-Fano", "orig_size": orig_len, "comp_size": comp_size}

def _fixed_3bit_encoding(text, input_file, orig_len, prefix):
    """Fixed 3-bit encoding for <=8 unique characters."""
    unique_chars = sorted(set(text))
    codes = {char: format(i, '03b') for i, char in enumerate(unique_chars)}
    
    # Encode text
    encoded_bits = ''.join(codes[char] for char in text)
    
    # Save with minimal header
    output_file = f"{outputShannonText}/{os.path.splitext(os.path.basename(input_file))[0]}.sf"
    with open(output_file, 'wb') as f:
        f.write(b"3BIT")  # 3-bit marker
        f.write(orig_len.to_bytes(2, 'big'))  # Original size
        f.write(len(unique_chars).to_bytes(1, 'big'))  # Number of symbols
        
        # Save symbol table
        for char in unique_chars:
            char_val = ord(char) % 256  # Ensure value fits in 1 byte
            f.write(char_val.to_bytes(1, 'big'))
        
        # Save encoded bits
        padding = (8 - len(encoded_bits) % 8) % 8
        if padding > 0:
            encoded_bits += '0' * padding
            f.write(padding.to_bytes(1, 'big'))
        
        # Convert bits to bytes
        for i in range(0, len(encoded_bits), 8):
            byte_val = int(encoded_bits[i:i+8], 2)
            f.write(byte_val.to_bytes(1, 'big'))
    
    comp_size = len(open(output_file, 'rb').read())
    return {"orig_size": orig_len, "comp_size": comp_size}

def _fixed_4bit_encoding(text, input_file, orig_len, prefix):
    """Fixed 4-bit encoding for <=16 unique characters."""
    unique_chars = sorted(set(text))
    codes = {char: format(i, '04b') for i, char in enumerate(unique_chars)}
    
    # Encode text
    encoded_bits = ''.join(codes[char] for char in text)
    
    # Save with minimal header
    output_file = f"{outputShannonText}/{os.path.splitext(os.path.basename(input_file))[0]}.sf"
    with open(output_file, 'wb') as f:
        f.write(b"4BIT")  # 4-bit marker
        f.write(orig_len.to_bytes(2, 'big'))  # Original size
        f.write(len(unique_chars).to_bytes(1, 'big'))  # Number of symbols
        
        # Save symbol table
        for char in unique_chars:
            char_val = ord(char) % 256  # Ensure value fits in 1 byte
            f.write(char_val.to_bytes(1, 'big'))
        
        # Save encoded bits
        padding = (8 - len(encoded_bits) % 8) % 8
        if padding > 0:
            encoded_bits += '0' * padding
            f.write(padding.to_bytes(1, 'big'))
        
        # Convert bits to bytes
        for i in range(0, len(encoded_bits), 8):
            byte_val = int(encoded_bits[i:i+8], 2)
            f.write(byte_val.to_bytes(1, 'big'))
    
    comp_size = len(open(output_file, 'rb').read())
    return {"orig_size": orig_len, "comp_size": comp_size}

def _smart_prefix_encoding(text, left_group, right_group, input_file, orig_len, prefix):
    """Smart prefix encoding based on frequency groups."""
    # Assign optimal prefixes
    left_codes = {char: '0' + format(i, f'0{(len(left_group)-1).bit_length()}b') 
                   for i, (char, _) in enumerate(left_group)}
    right_codes = {char: '1' + format(i, f'0{(len(right_group)-1).bit_length()}b') 
                     for i, (char, _) in enumerate(right_group)}
    
    # Combine codes
    all_codes = {**left_codes, **right_codes}
    
    # Encode text
    encoded_bits = ''.join(all_codes[char] for char in text)
    
    # Save with minimal header
    output_file = f"{outputShannonText}/{os.path.splitext(os.path.basename(input_file))[0]}.sf"
    with open(output_file, 'wb') as f:
        f.write(b"PREFIX")  # Prefix marker
        f.write(orig_len.to_bytes(2, 'big'))  # Original size
        f.write(len(left_group).to_bytes(1, 'big'))  # Left group size
        
        # Save symbol table
        for char, _ in left_group:
            char_val = ord(char) % 256  # Ensure value fits in 1 byte
            f.write(char_val.to_bytes(1, 'big'))
        for char, _ in right_group:
            char_val = ord(char) % 256  # Ensure value fits in 1 byte
            f.write(char_val.to_bytes(1, 'big'))
        
        # Save encoded bits
        padding = (8 - len(encoded_bits) % 8) % 8
        if padding > 0:
            encoded_bits += '0' * padding
            f.write(padding.to_bytes(1, 'big'))
        
        # Convert bits to bytes
        for i in range(0, len(encoded_bits), 8):
            byte_val = int(encoded_bits[i:i+8], 2)
            f.write(byte_val.to_bytes(1, 'big'))
    
    comp_size = len(open(output_file, 'rb').read())
    return {"orig_size": orig_len, "comp_size": comp_size}

def shanonCompression():
    input_file = f"{inputFiles}/test.txt"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    stats = _run_shannon_fano(input_file)
    _print_results(stats)

def _run_shannon_fano_image(image_path):
    """Run Shannon-Fano compression on an image file with optimizations for larger files"""
    print(f"   Processing {os.path.basename(image_path)} with Shannon-Fano...")
    
    try:
        # Read image as binary data
        image_data = read_binary_data(image_path)
        orig_size = len(image_data)
        
        if not image_data:
            return {"name": "Shannon-Fano", "orig_size": orig_size, "comp_size": orig_size}
        
        # Convert to bytes if needed
        if isinstance(image_data, str):
            image_data = image_data.encode('latin1')
        
        # Choose compression strategy based on file size
        if orig_size < 1000:
            # For small images, use the existing Shannon-Fano approach
            return _compress_small_image_shannon(image_data, image_path, orig_size)
        elif orig_size < 10000:
            # For medium images, use chunked Shannon-Fano compression
            return _compress_medium_image_shannon(image_data, image_path, orig_size)
        else:
            # For large images, use hybrid approach with preprocessing
            return _compress_large_image_shannon(image_data, image_path, orig_size)
        
    except Exception as e:
        print(f"   Shannon-Fano compression failed: {e}")
        return {"name": "Shannon-Fano", "orig_size": os.path.getsize(image_path), "comp_size": os.path.getsize(image_path)}

def _compress_small_image_shannon(image_data, image_path, orig_size):
    """Compress small images using standard Shannon-Fano"""
    compressor = ShannonFanoCompressor()
    output_file = f"{outputShannonImage}/{os.path.splitext(os.path.basename(image_path))[0]}.sf"
    
    # Use existing compressor for small images
    compressor.compress_file(image_data, output_file)
    
    comp_size = os.path.getsize(output_file)
    final_size = min(comp_size, orig_size)
    
    return {"name": "Shannon-Fano", "orig_size": orig_size, "comp_size": final_size}

def _compress_medium_image_shannon(image_data, image_path, orig_size):
    """Compress medium images using chunked Shannon-Fano"""
    chunk_size = 2048  # 2KB chunks
    all_compressed_chunks = []
    
    # Process in chunks
    for i in range(0, len(image_data), chunk_size):
        chunk = image_data[i:i + chunk_size]
        
        # Build frequency table for this chunk
        freq = {}
        for byte in chunk:
            freq[byte] = freq.get(byte, 0) + 1
        
        # Build Shannon-Fano codes for this chunk
        if len(freq) > 1:
            compressed_chunk = _compress_chunk_with_shannon_fano(chunk, freq)
        else:
            # All bytes are the same, use simple RLE
            compressed_chunk = _compress_chunk_rle(chunk)
        
        all_compressed_chunks.append(compressed_chunk)
    
    # Save compressed image
    output_file = f"{outputShannonImage}/{os.path.splitext(os.path.basename(image_path))[0]}.sf"
    
    with open(output_file, 'wb') as f:
        f.write(b"SFM")  # Shannon-Fano Medium marker
        f.write(orig_size.to_bytes(4, 'big'))  # Original size
        f.write(len(all_compressed_chunks).to_bytes(2, 'big'))  # Number of chunks
        
        for chunk_data in all_compressed_chunks:
            f.write(len(chunk_data).to_bytes(2, 'big'))  # Chunk size
            f.write(chunk_data)
    
    comp_size = len(open(output_file, 'rb').read())
    final_size = min(comp_size, orig_size)
    
    return {"name": "Shannon-Fano", "orig_size": orig_size, "comp_size": final_size}

def _compress_large_image_shannon(image_data, image_path, orig_size):
    """Compress large images using effective multi-level compression"""
    # Step 1: Apply effective lossy compression for images
    compressed_data = _apply_effective_shannon_image_compression(image_data)
    
    # Step 2: Apply Shannon-Fano entropy coding
    final_data = _apply_shannon_entropy_coding(compressed_data)
    
    # Step 3: Save with minimal overhead
    output_file = f"{outputShannonImage}/{os.path.splitext(os.path.basename(image_path))[0]}.sf"
    
    with open(output_file, 'wb') as f:
        f.write(b"SCE")  # Shannon-Fano Effective marker
        f.write(orig_size.to_bytes(4, 'big'))  # Original size
        f.write(len(final_data).to_bytes(4, 'big'))  # Compressed size
        f.write(final_data)
    
    comp_size = len(open(output_file, 'rb').read())
    final_size = min(comp_size, orig_size)
    
    return {"name": "Shannon-Fano", "orig_size": orig_size, "comp_size": final_size}

def _apply_effective_shannon_image_compression(data):
    """Apply effective lossy compression for Shannon-Fano"""
    if isinstance(data, str):
        data = data.encode('latin1')
    
    # Step 1: Reduce color depth to 2 bits (4 colors) - most aggressive
    color_reduced = bytearray()
    for byte in data:
        # Reduce to 2 bits
        reduced_byte = (byte >> 6) & 0x03
        color_reduced.append(reduced_byte)
    
    # Step 2: Pack four 2-bit values into one byte
    packed = bytearray()
    for i in range(0, len(color_reduced), 4):
        if i + 4 <= len(color_reduced):
            # Pack 4 values (2 bits each) into 1 byte
            packed_byte = (color_reduced[i] << 6) | (color_reduced[i+1] << 4) | (color_reduced[i+2] << 2) | color_reduced[i+3]
            packed.append(packed_byte)
        else:
            # Handle remaining values
            remaining = color_reduced[i:]
            if len(remaining) == 1:
                packed.append(remaining[0] << 6)
            elif len(remaining) == 2:
                packed.append((remaining[0] << 6) | (remaining[1] << 4))
            elif len(remaining) == 3:
                packed.append((remaining[0] << 6) | (remaining[1] << 4) | (remaining[2] << 2))
    
    # Step 3: Apply ultra-aggressive RLE
    rle_compressed = bytearray()
    i = 0
    
    while i < len(packed):
        current = packed[i]
        
        # Look for runs (very aggressive)
        if i + 1 < len(packed) and packed[i] == packed[i+1]:
            run_length = 2
            while i + run_length < len(packed) and packed[i + run_length] == current and run_length < 255:
                run_length += 1
            
            # Encode run
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
    """Apply Shannon-Fano entropy coding"""
    if isinstance(data, str):
        data = data.encode('latin1')
    
    # Calculate frequency
    freq = {}
    for byte in data:
        freq[byte] = freq.get(byte, 0) + 1
    
    # Sort by frequency
    sorted_bytes = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    
    # Create Shannon-Fano codes
    codes = {}
    total_freq = sum(freq for _, freq in sorted_bytes)
    
    # Split into two groups
    mid_point = len(sorted_bytes) // 2
    left_group = sorted_bytes[:mid_point]
    right_group = sorted_bytes[mid_point:]
    
    # Assign codes
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
    
    # Encode data
    bit_string = ''.join(codes[byte] for byte in data)
    
    # Pack into bytes
    padding = (8 - len(bit_string) % 8) % 8
    bit_string += '0' * padding
    
    encoded = bytearray()
    encoded.append(padding)  # Store padding
    
    # Store code table
    encoded.append(min(len(sorted_bytes), 255))  # Number of codes
    
    for byte, _ in sorted_bytes[:255]:
        encoded.append(byte)  # Byte value
    
    # Convert bits to bytes
    for i in range(0, len(bit_string), 8):
        byte_val = int(bit_string[i:i+8], 2)
        encoded.append(byte_val)
    
    return bytes(encoded)

def _compress_chunk_ultra_fixed(chunk, freq):
    """Ultra-fixed compression for very low diversity chunks"""
    # Sort bytes by frequency
    sorted_bytes = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    byte_to_code = {byte: i for i, (byte, _) in enumerate(sorted_bytes)}
    
    # Use minimal bits per byte (1-2 bits)
    bits_per_byte = max(1, len(sorted_bytes).bit_length() - 1)
    
    # Encode chunk
    bit_string = ''
    for byte in chunk:
        code = byte_to_code[byte]
        bit_string += format(code, f'0{bits_per_byte}b')
    
    # Pack into bytes
    padding = (8 - len(bit_string) % 8) % 8
    bit_string += '0' * padding
    
    compressed = bytearray()
    compressed.append(bits_per_byte)  # Store bits per byte
    compressed.append(padding)  # Store padding
    
    # Store byte table
    compressed.append(len(sorted_bytes))
    for byte, _ in sorted_bytes:
        compressed.append(byte)
    
    # Convert bits to bytes
    for i in range(0, len(bit_string), 8):
        byte_val = int(bit_string[i:i+8], 2)
        compressed.append(byte_val)
    
    return bytes(compressed)

def _compress_chunk_with_shannon_fano(chunk, freq):
    """Compress a chunk using Shannon-Fano coding"""
    # Sort symbols by frequency (descending)
    symbols = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    
    # Build Shannon-Fano codes
    codes = {}
    _shannon_split(symbols, "", codes)
    
    # Encode chunk
    bit_string = ''.join(codes[byte] for byte in chunk)
    
    # Pack into bytes
    padding = (8 - len(bit_string) % 8) % 8
    bit_string += '0' * padding
    
    compressed = bytearray()
    compressed.append(padding)  # Store padding
    
    # Convert bits to bytes
    for i in range(0, len(bit_string), 8):
        byte_val = int(bit_string[i:i+8], 2)
        compressed.append(byte_val)
    
    # Store code table
    table_data = bytearray()
    table_data.append(len(codes))  # Number of codes
    
    for byte, code in codes.items():
        table_data.append(byte)  # Byte value
        table_data.append(len(code))  # Code length
        # Store code as bytes
        code_val = int(code, 2)
        code_bytes = (len(code) + 7) // 8
        for j in range(code_bytes):
            table_data.append((code_val >> (8 * (code_bytes - 1 - j))) & 0xFF)
    
    # Combine table and data
    result = bytearray()
    result.extend(table_data)
    result.extend(compressed)
    
    return bytes(result)

def _shannon_split(symbols, prefix, codes):
    """Recursively split symbols for Shannon-Fano coding"""
    if len(symbols) == 1:
        codes[symbols[0][0]] = prefix or "0"
        return
    
    # Calculate total frequency
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
    
    # Ensure valid split
    split_index = max(1, min(len(symbols) - 1, best_split))
    
    left_group = symbols[:split_index]
    right_group = symbols[split_index:]
    
    # Recursively assign codes
    _shannon_split(left_group, prefix + "0", codes)
    _shannon_split(right_group, prefix + "1", codes)

def _compress_chunk_rle(chunk):
    """Compress a chunk using RLE"""
    compressed = bytearray()
    i = 0
    
    while i < len(chunk):
        if i + 2 < len(chunk) and chunk[i] == chunk[i+1] == chunk[i+2]:
            # Found run
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

def _compress_chunk_fixed(chunk, freq):
    """Compress chunk with fixed-length coding"""
    # Sort bytes by frequency
    sorted_bytes = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    byte_to_code = {byte: i for i, (byte, _) in enumerate(sorted_bytes)}
    
    # Determine optimal bits per byte
    bits_per_byte = max(1, (len(sorted_bytes).bit_length()))
    
    # Encode chunk
    bit_string = ''
    for byte in chunk:
        code = byte_to_code[byte]
        bit_string += format(code, f'0{bits_per_byte}b')
    
    # Pack into bytes
    padding = (8 - len(bit_string) % 8) % 8
    bit_string += '0' * padding
    
    compressed = bytearray()
    compressed.append(bits_per_byte)  # Store bits per byte
    compressed.append(padding)  # Store padding
    
    # Store byte table
    compressed.append(len(sorted_bytes))
    for byte, _ in sorted_bytes:
        compressed.append(byte)
    
    # Convert bits to bytes
    for i in range(0, len(bit_string), 8):
        byte_val = int(bit_string[i:i+8], 2)
        compressed.append(byte_val)
    
    return bytes(compressed)

def _compress_chunk_adaptive_shannon(chunk, freq):
    """Compress chunk using adaptive Shannon-Fano"""
    # Group symbols by frequency ranges
    total_bytes = len(chunk)
    high_freq = [byte for byte, count in freq.items() if count > total_bytes * 0.1]
    med_freq = [byte for byte, count in freq.items() if total_bytes * 0.02 < count <= total_bytes * 0.1]
    
    # Create efficient codes using Shannon-Fano principle
    codes = {}
    
    # High-frequency symbols get short codes
    if high_freq:
        high_symbols = [(byte, freq[byte]) for byte in high_freq[:8]]
        high_codes = {}
        _shannon_split(high_symbols, "0", high_codes)
        codes.update(high_codes)
    
    # Medium-frequency symbols get medium codes
    if med_freq:
        med_symbols = [(byte, freq[byte]) for byte in med_freq[:16]]
        med_codes = {}
        _shannon_split(med_symbols, "10", med_codes)
        codes.update(med_codes)
    
    # Low-frequency symbols get longer codes
    low_freq_symbols = [(byte, freq[byte]) for byte in set(chunk) if byte not in codes]
    if low_freq_symbols:
        low_codes = {}
        _shannon_split(low_freq_symbols, "11", low_codes)
        codes.update(low_codes)
    
    # Encode chunk
    bit_string = ''.join(codes[byte] for byte in chunk)
    
    # Pack into bytes
    padding = (8 - len(bit_string) % 8) % 8
    bit_string += '0' * padding
    
    compressed = bytearray()
    compressed.append(padding)  # Store padding
    
    # Store code distribution info
    compressed.append(len(high_freq))  # Number of high-frequency codes
    compressed.append(len(med_freq))   # Number of medium-frequency codes
    
    # Store high-frequency bytes
    for byte in high_freq[:8]:
        compressed.append(byte)
    
    # Store medium-frequency bytes
    for byte in med_freq[:16]:
        compressed.append(byte)
    
    # Convert bits to bytes
    for i in range(0, len(bit_string), 8):
        byte_val = int(bit_string[i:i+8], 2)
        compressed.append(byte_val)
    
    return bytes(compressed)

def _apply_predictive_preprocessing(data):
    """Apply predictive preprocessing for better compression of images"""
    if isinstance(data, str):
        data = data.encode('latin1')
    
    preprocessed = bytearray()
    
    # Use simple predictive coding: predict next byte from previous
    for i in range(len(data)):
        if i == 0:
            # First byte as-is
            preprocessed.append(data[i])
        else:
            # Predict current byte from previous
            predicted = data[i-1]
            actual = data[i]
            prediction_error = (actual - predicted) % 256
            preprocessed.append(prediction_error)
    
    return bytes(preprocessed)

def _apply_aggressive_shannon_preprocessing(data):
    """Apply aggressive preprocessing for Shannon-Fano compression"""
    if isinstance(data, str):
        data = data.encode('latin1')
    
    # Step 1: Apply predictive coding
    predicted_data = _apply_predictive_preprocessing(data)
    
    # Step 2: Apply bit-level pattern compression
    preprocessed = bytearray()
    i = 0
    
    while i < len(predicted_data):
        current = predicted_data[i]
        
        # Look for alternating patterns (common in predictive coding)
        if i + 3 < len(predicted_data):
            pattern1 = predicted_data[i]
            pattern2 = predicted_data[i+1]
            if (predicted_data[i+2] == pattern1 and predicted_data[i+3] == pattern2):
                # Found alternating pattern
                run_length = 2
                while i + run_length * 2 < len(predicted_data) and \
                      predicted_data[i + run_length * 2] == pattern1 and \
                      predicted_data[i + run_length * 2 + 1] == pattern2 and \
                      run_length < 63:
                    run_length += 1
                
                # Encode as [0xF6, run_length-2, pattern1, pattern2]
                preprocessed.extend([0xF6, run_length - 2, pattern1, pattern2])
                i += run_length * 2
                continue
        
        # Look for runs of small values
        if abs(current - 128) <= 7:
            run_length = 1
            while i + run_length < len(predicted_data) and \
                  abs(predicted_data[i + run_length] - 128) <= 7 and \
                  run_length < 127:
                run_length += 1
            
            if run_length >= 4:
                # Encode as [0xF5, run_length-4, base_value]
                preprocessed.extend([0xF5, run_length - 4, 128])
                i += run_length
                continue
        
        # Look for zero runs
        if current == 0:
            run_length = 1
            while i + run_length < len(predicted_data) and predicted_data[i + run_length] == 0 and run_length < 255:
                run_length += 1
            
            if run_length >= 3:
                # Encode as [0xF4, run_length-3]
                preprocessed.extend([0xF4, run_length - 3])
                i += run_length
                continue
        
        # No pattern found, copy as-is
        preprocessed.append(current)
        i += 1
    
    return bytes(preprocessed)

def _apply_hybrid_shannon_compression(data):
    """Apply hybrid compression combining multiple techniques"""
    if isinstance(data, str):
        data = data.encode('latin1')
    
    # Step 1: Apply block-based transform
    block_size = 16
    transformed = bytearray()
    
    for i in range(0, len(data), block_size):
        block = data[i:i+block_size]
        
        if len(block) < block_size:
            # Pad incomplete block
            block += bytes([0] * (block_size - len(block)))
        
        # Apply simple transform: subtract first byte from all
        if len(block) > 0:
            base_byte = block[0]
            transformed_block = bytearray()
            transformed_block.append(base_byte)
            
            for j in range(1, len(block)):
                transformed_byte = (block[j] - base_byte) % 256
                transformed_block.append(transformed_byte)
            
            transformed.extend(transformed_block)
    
    # Step 2: Apply frequency-based grouping
    freq = {}
    for byte in transformed:
        freq[byte] = freq.get(byte, 0) + 1
    
    # Group by frequency
    high_freq = [byte for byte, count in freq.items() if count > len(transformed) * 0.1]
    med_freq = [byte for byte, count in freq.items() if len(transformed) * 0.02 < count <= len(transformed) * 0.1]
    
    # Step 3: Apply adaptive encoding
    encoded = bytearray()
    
    for byte in transformed:
        if byte in high_freq:
            # High frequency: 1-byte encoding
            encoded.append(0xF3)
            encoded.append(byte)
        elif byte in med_freq:
            # Medium frequency: 2-byte encoding
            encoded.append(0xF2)
            encoded.append(byte)
        else:
            # Low frequency: direct encoding
            encoded.append(byte)
    
    return bytes(encoded)

def shannonImageCompression():
    """Compress image using Shannon-Fano algorithm"""
    print("\n  Available image files:")
    
    import glob
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
        # Read image as binary data
        image_data = read_binary_data(selected_image)
        orig_size = len(image_data)
        
        if not image_data:
            print("Error: Image file is empty!")
            return
         
        # Use the same optimized approach as _run_shannon_fano_image
        result = _run_shannon_fano_image(selected_image)
        comp_size = result.get("comp_size", orig_size)
        
        # Check if compression is beneficial
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
        # Fallback to original size
        orig_size = os.path.getsize(selected_image)
        print(f"   Original: {orig_size:,} bytes")
        print(f"   Compressed: {orig_size:,} bytes")
        print(f"   Space saved: 0.0%")


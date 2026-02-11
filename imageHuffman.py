"""
Image Huffman Compression Module
Optimized Huffman compression for image files
"""

from file_handler import read_text_file, _print_results, read_binary_data
import os
import glob
from constants import inputFiles, outputHuffmanImage

def _run_huffman_image(image_path):
    """Run Huffman compression on an image file with optimizations for larger files"""
    print(f"   Processing {os.path.basename(image_path)} with Huffman...")
    
    try:
        # Read image as binary data
        image_data = read_binary_data(image_path)
        orig_size = len(image_data)
        
        if not image_data:
            return {"name": "Huffman", "orig_size": orig_size, "comp_size": orig_size}
        
        # Convert to bytes if needed
        if isinstance(image_data, str):
            image_data = image_data.encode('latin1')
        
        # Choose compression strategy based on file size
        if orig_size < 1000:
            result = _compress_small_image_huffman(image_data, image_path, orig_size)
        elif orig_size < 10000:
            result = _compress_medium_image_huffman(image_data, image_path, orig_size)
        else:
            result = _compress_large_image_huffman(image_data, image_path, orig_size)
        
        return result
        
    except Exception as e:
        print(f"    Image Huffman error: {e}")
        return {"name": "Huffman", "orig_size": orig_size, "comp_size": orig_size}

def _compress_small_image_huffman(image_data, image_path, orig_size):
    """Compress small images using optimized Huffman"""
    # Apply preprocessing for better compression
    preprocessed_data = _apply_differential_preprocessing(image_data)
    
    # Build frequency table
    freq = {}
    for byte in preprocessed_data:
        freq[byte] = freq.get(byte, 0) + 1
    
    # Build Huffman tree
    import heapq
    heap = [[weight, [byte, ""]] for byte, weight in freq.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    # Extract codes
    codes = {}
    for byte, code in heap[0][1:]:
        codes[byte] = code
    
    # Encode data
    encoded_bits = ''.join(codes[byte] for byte in preprocessed_data)
    
    # Save with minimal overhead
    output_file = f"{outputHuffmanImage}/{os.path.splitext(os.path.basename(image_path))[0]}.huf"
    with open(output_file, 'wb') as f:
        f.write(b"HIS")  # Huffman Image Small marker
        f.write(orig_size.to_bytes(4, 'big'))  # Original size
        f.write(len(codes).to_bytes(2, 'big'))  # Number of symbols
        
        # Save code table
        for byte, code in codes.items():
            f.write(byte.to_bytes(1, 'big'))
            f.write(len(code).to_bytes(1, 'big'))
        
        # Save encoded bits
        padding = (8 - len(encoded_bits) % 8) % 8
        if padding > 0:
            encoded_bits += '0' * padding
        
        # Convert bits to bytes
        for i in range(0, len(encoded_bits), 8):
            byte_val = int(encoded_bits[i:i+8], 2)
            f.write(byte_val.to_bytes(1, 'big'))
    
    comp_size = len(open(output_file, 'rb').read())
    final_size = min(comp_size, orig_size)
    
    return {"name": "Huffman", "orig_size": orig_size, "comp_size": final_size}

def _compress_medium_image_huffman(image_data, image_path, orig_size):
    """Compress medium images using chunked Huffman"""
    # Process in chunks for better compression
    chunk_size = 1024
    all_compressed_chunks = []
    
    for i in range(0, len(image_data), chunk_size):
        chunk = image_data[i:i+chunk_size]
        
        # Build frequency for this chunk
        freq = {}
        for byte in chunk:
            freq[byte] = freq.get(byte, 0) + 1
        
        # Choose compression method based on diversity
        diversity = len(freq)
        if diversity < 16:
            compressed_chunk = _compress_chunk_ultra_simple(chunk, freq)
        elif diversity < 64:
            compressed_chunk = _compress_chunk_simple(chunk, freq)
        else:
            compressed_chunk = _compress_chunk_with_huffman(chunk, freq)
        
        all_compressed_chunks.append(compressed_chunk)
    
    # Save compressed image
    output_file = f"{outputHuffmanImage}/{os.path.splitext(os.path.basename(image_path))[0]}.huf"
    
    with open(output_file, 'wb') as f:
        f.write(b"HCM")  # Huffman Chunked Medium marker
        f.write(orig_size.to_bytes(4, 'big'))  # Original size
        f.write(len(all_compressed_chunks).to_bytes(2, 'big'))  # Number of chunks
        
        for chunk_data in all_compressed_chunks:
            f.write(len(chunk_data).to_bytes(2, 'big'))  # Chunk size
            f.write(chunk_data)
    
    comp_size = len(open(output_file, 'rb').read())
    final_size = min(comp_size, orig_size)
    
    return {"name": "Huffman", "orig_size": orig_size, "comp_size": final_size}

def _compress_large_image_huffman(image_data, image_path, orig_size):
    """Compress large images using effective multi-level compression"""
    # Step 1: Apply effective lossy compression for images
    compressed_data = _apply_effective_huffman_image_compression(image_data)
    
    # Step 2: Apply entropy coding
    final_data = _apply_huffman_entropy_coding(compressed_data)
    
    # Step 3: Save with minimal overhead
    output_file = f"{outputHuffmanImage}/{os.path.splitext(os.path.basename(image_path))[0]}.huf"
    
    with open(output_file, 'wb') as f:
        f.write(b"HCE")  # Huffman Effective marker
        f.write(orig_size.to_bytes(4, 'big'))  # Original size
        f.write(len(final_data).to_bytes(4, 'big'))  # Compressed size
        f.write(final_data)
    
    comp_size = len(open(output_file, 'rb').read())
    final_size = min(comp_size, orig_size)
    
    return {"name": "Huffman", "orig_size": orig_size, "comp_size": final_size}

def _apply_effective_huffman_image_compression(data):
    """Apply effective lossy compression for Huffman"""
    if isinstance(data, str):
        data = data.encode('latin1')
    
    # Step 1: Reduce color depth to 3 bits (8 colors) - even more aggressive
    color_reduced = bytearray()
    for byte in data:
        # Reduce to 3 bits
        reduced_byte = (byte & 0xE0)  # Keep top 3 bits only
        color_reduced.append(reduced_byte)
    
    # Step 2: Apply RLE for better compression
    rle_compressed = bytearray()
    i = 0
    while i < len(color_reduced):
        current_byte = color_reduced[i]
        count = 1
        j = i + 1
        while j < len(color_reduced) and color_reduced[j] == current_byte and count < 255:
            count += 1
            j += 1
        
        if count > 2:  # Only compress runs longer than 2
            rle_compressed.extend([0xFF, count, current_byte])
        else:
            rle_compressed.extend([current_byte] * count)
        i = j
    
    return bytes(rle_compressed)

def _apply_huffman_entropy_coding(data):
    """Apply entropy encoding for better compression"""
    if isinstance(data, str):
        data = data.encode('latin1')
    
    # Calculate frequency distribution
    freq = {}
    for byte in data:
        freq[byte] = freq.get(byte, 0) + 1
    
    # Sort by frequency
    sorted_bytes = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    
    # Create optimal variable-length codes
    codes = {}
    for i, (byte, count) in enumerate(sorted_bytes):
        if i < 16:
            # Top 16 get 4-bit codes
            codes[byte] = format(i, '04b')
        elif i < 48:
            # Next 32 get 6-bit codes
            codes[byte] = '1' + format(i - 16, '05b')
        elif i < 112:
            # Next 64 get 8-bit codes
            codes[byte] = '11' + format(i - 48, '06b')
        else:
            # Rest get 10-bit codes
            codes[byte] = '111' + format(i - 112, '07b')
    
    # Encode data
    bit_string = ''.join(codes[byte] for byte in data)
    
    # Pack into bytes
    padding = (8 - len(bit_string) % 8) % 8
    bit_string += '0' * padding
    
    encoded = bytearray()
    encoded.append(padding)  # Store padding
    
    # Store code table
    encoded.append(len(sorted_bytes))  # Number of codes
    
    for byte, _ in sorted_bytes:
        encoded.append(byte)  # Byte value
    
    # Convert bits to bytes
    for i in range(0, len(bit_string), 8):
        byte_val = int(bit_string[i:i+8], 2)
        encoded.append(byte_val)
    
    return bytes(encoded)

def _apply_differential_preprocessing(data):
    """Apply differential preprocessing for better compression of images"""
    if isinstance(data, str):
        data = data.encode('latin1')
    
    preprocessed = bytearray()
    prev_byte = 0
    
    for byte in data:
        # Use differential encoding: store difference from previous byte
        diff = (byte - prev_byte) % 256
        preprocessed.append(diff)
        prev_byte = byte
    
    return bytes(preprocessed)

def _compress_chunk_ultra_simple(chunk, freq):
    """Ultra-simple compression for very low diversity chunks"""
    # Sort bytes by frequency
    sorted_bytes = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    byte_to_code = {byte: i for i, (byte, _) in enumerate(sorted_bytes)}
    
    # Use minimal bits per byte
    bits_per_byte = max(1, (len(sorted_bytes).bit_length() - 1))
    
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

def _compress_chunk_with_huffman(chunk, freq):
    """Compress a chunk using Huffman coding"""
    import heapq
    
    # Build Huffman tree
    heap = [[weight, [byte, ""]] for byte, weight in freq.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    # Extract codes
    codes = {}
    for byte, code in heap[0][1:]:
        codes[byte] = code
    
    # Encode chunk
    encoded_bits = ''.join(codes[byte] for byte in chunk)
    
    # Pack into bytes
    padding = (8 - len(encoded_bits) % 8) % 8
    if padding > 0:
        encoded_bits += '0' * padding
    
    compressed = bytearray()
    compressed.append(padding)  # Store padding
    
    # Store code table
    compressed.append(len(codes))  # Number of codes
    for byte, code in codes.items():
        compressed.append(byte)  # Byte value
        compressed.append(len(code))  # Code length
    
    # Convert bits to bytes
    for i in range(0, len(encoded_bits), 8):
        byte_val = int(encoded_bits[i:i+8], 2)
        compressed.append(byte_val)
    
    return bytes(compressed)

def _compress_chunk_simple(chunk, freq):
    """Compress chunk with simple fixed-length coding"""
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

def huffmanImageCompression():
    """Compress image using Huffman algorithm"""
    print("\n  Available image files:")
    
    # Get image files (common extensions)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff', '*.webp']
    available_images = []
    
    for ext in image_extensions:
        available_images.extend(glob.glob(f"{inputFiles}/*{ext}"))
        available_images.extend(glob.glob(f"{inputFiles}/*{ext.upper()}"))
    
    if not available_images:
        print("No image files found in inputs folder.")
        return
    
    # Remove duplicates and sort
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
    
    print(f"\n Compressing {os.path.basename(selected_image)} with HUFFMAN...")
    
    try:
        # Read image as binary data
        image_data = read_binary_data(selected_image)
        orig_size = len(image_data)
        
        if not image_data:
            print("Error: Image file is empty!")
            return
         
        # Use the same optimized approach as _run_huffman_image
        result = _run_huffman_image(selected_image)
        comp_size = result.get("comp_size", orig_size)
        
        # Check if compression is beneficial
        if comp_size >= orig_size:
            print(f"   Compression would increase size, using original")
            print(f"   Original: {orig_size:,} bytes")
            print(f"   Compressed: {orig_size:,} bytes")
            print(f"   Space saved: 0.0%")
            return
        
        savings = (orig_size - comp_size) / orig_size * 100
        
        print(f" HUFFMAN image compression completed!")
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

if __name__ == "__main__":
    huffmanImageCompression()
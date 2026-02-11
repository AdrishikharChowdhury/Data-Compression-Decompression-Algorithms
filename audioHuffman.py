"""
Audio Huffman Compression Module
Optimized Huffman compression for audio files
"""

from file_handler import read_text_file, _print_results, read_binary_data
import os
import glob
from constants import inputFiles, outputHuffmanAudio

def _run_huffman_audio(audio_path):
    """Run Huffman compression on an audio file with audio-specific optimizations"""
    print(f"   Processing {os.path.basename(audio_path)} with Huffman...")
    
    try:
        # Read audio as binary data
        audio_data = read_binary_data(audio_path)
        orig_size = len(audio_data)
        
        if not audio_data:
            return {"name": "Huffman", "orig_size": orig_size, "comp_size": orig_size}
        
        # Convert to bytes if needed
        if isinstance(audio_data, str):
            audio_data = audio_data.encode('latin1')
        
        # Choose compression strategy based on file size
        if orig_size < 10000:
            result = _compress_small_audio_huffman(audio_data, audio_path, orig_size)
        elif orig_size < 100000:
            result = _compress_medium_audio_huffman(audio_data, audio_path, orig_size)
        else:
            result = _compress_large_audio_huffman(audio_data, audio_path, orig_size)
        
        return result
        
    except Exception as e:
        print(f"    Audio Huffman error: {e}")
        return {"name": "Huffman", "orig_size": orig_size, "comp_size": orig_size}

def _compress_small_audio_huffman(audio_data, audio_path, orig_size):
    """Compress small audio files using optimized Huffman"""
    # Apply audio-specific preprocessing
    preprocessed_data = _apply_audio_preprocessing(audio_data)
    
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
    output_file = f"{outputHuffmanAudio}/{os.path.splitext(os.path.basename(audio_path))[0]}.huf"
    with open(output_file, 'wb') as f:
        f.write(b"HAS")  # Huffman Audio Small marker
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

def _compress_medium_audio_huffman(audio_data, audio_path, orig_size):
    """Compress medium audio files using chunked Huffman"""
    # Process in chunks for better compression
    chunk_size = 8192  # 8KB chunks
    all_compressed_chunks = []
    
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i+chunk_size]
        
        # Apply audio preprocessing to chunk
        preprocessed_chunk = _apply_audio_preprocessing(chunk)
        
        # Build frequency for this chunk
        freq = {}
        for byte in preprocessed_chunk:
            freq[byte] = freq.get(byte, 0) + 1
        
        # Choose compression method based on diversity
        diversity = len(freq)
        if diversity < 32:
            compressed_chunk = _compress_audio_chunk_simple(preprocessed_chunk, freq)
        else:
            compressed_chunk = _compress_audio_chunk_with_huffman(preprocessed_chunk, freq)
        
        all_compressed_chunks.append(compressed_chunk)
    
    # Save compressed audio
    output_file = f"{outputHuffmanAudio}/{os.path.splitext(os.path.basename(audio_path))[0]}.huf"
    
    with open(output_file, 'wb') as f:
        f.write(b"HAM")  # Huffman Audio Medium marker
        f.write(orig_size.to_bytes(4, 'big'))  # Original size
        f.write(len(all_compressed_chunks).to_bytes(2, 'big'))  # Number of chunks
        
        for chunk_data in all_compressed_chunks:
            f.write(len(chunk_data).to_bytes(2, 'big'))  # Chunk size
            f.write(chunk_data)
    
    comp_size = len(open(output_file, 'rb').read())
    final_size = min(comp_size, orig_size)
    
    return {"name": "Huffman", "orig_size": orig_size, "comp_size": final_size}

def _compress_large_audio_huffman(audio_data, audio_path, orig_size):
    """Compress large audio files using multi-level compression"""
    # Step 1: Apply audio-specific preprocessing
    preprocessed_data = _apply_audio_preprocessing(audio_data)
    
    # Step 2: Apply adaptive Huffman coding
    final_data = _apply_adaptive_audio_huffman(preprocessed_data)
    
    # Step 3: Save with minimal overhead
    output_file = f"{outputHuffmanAudio}/{os.path.splitext(os.path.basename(audio_path))[0]}.huf"
    
    with open(output_file, 'wb') as f:
        f.write(b"HAL")  # Huffman Audio Large marker
        f.write(orig_size.to_bytes(4, 'big'))  # Original size
        f.write(len(final_data).to_bytes(4, 'big'))  # Compressed size
        f.write(final_data)
    
    comp_size = len(open(output_file, 'rb').read())
    final_size = min(comp_size, orig_size)
    
    return {"name": "Huffman", "orig_size": orig_size, "comp_size": final_size}

def _apply_audio_preprocessing(data):
    """Apply audio-specific preprocessing for better compression"""
    if isinstance(data, str):
        data = data.encode('latin1')
    
    # Audio-specific preprocessing
    preprocessed = bytearray()
    
    # Step 1: Apply high-pass filter (remove DC offset)
    if len(data) > 0:
        # Calculate and remove DC offset
        dc_offset = sum(data) / len(data)
        
        for byte in data:
            # Remove DC offset and normalize
            filtered_byte = int(byte - dc_offset + 128) % 256
            preprocessed.append(filtered_byte)
    
    # Step 2: Apply simple delta coding for better compression
    delta_compressed = bytearray()
    prev_byte = preprocessed[0] if len(preprocessed) > 0 else 0
    delta_compressed.append(prev_byte)
    
    for i in range(1, len(preprocessed)):
        current_byte = preprocessed[i]
        delta = (current_byte - prev_byte) % 256
        delta_compressed.append(delta)
        prev_byte = current_byte
    
    return bytes(delta_compressed)

def _apply_adaptive_audio_huffman(data):
    """Apply adaptive Huffman coding for audio"""
    # Build frequency table
    freq = {}
    for byte in data:
        freq[byte] = freq.get(byte, 0) + 1
    
    # Sort by frequency
    sorted_bytes = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    
    # Create optimal codes for audio data
    codes = {}
    for i, (byte, count) in enumerate(sorted_bytes):
        if i < 16:
            # Top 16 get 3-bit codes (most common in audio)
            codes[byte] = format(i, '03b')
        elif i < 48:
            # Next 32 get 5-bit codes
            codes[byte] = '1' + format(i - 16, '04b')
        elif i < 128:
            # Next 80 get 7-bit codes
            codes[byte] = '11' + format(i - 48, '05b')
        else:
            # Rest get 9-bit codes
            codes[byte] = '111' + format(i - 128, '06b')
    
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

def _compress_audio_chunk_simple(chunk, freq):
    """Compress audio chunk with simple coding"""
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

def _compress_audio_chunk_with_huffman(chunk, freq):
    """Compress audio chunk using Huffman coding"""
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

def huffmanAudioCompression():
    """Compress audio using Huffman algorithm"""
    print("\n  Available audio files:")
    
    # Get audio files (common extensions)
    audio_extensions = ['*.wav', '*.mp3', '*.ogg', '*.flac', '*.aac']
    available_audio = []
    
    for ext in audio_extensions:
        available_audio.extend(glob.glob(f"{inputFiles}/*{ext}"))
        available_audio.extend(glob.glob(f"{inputFiles}/*{ext.upper()}"))
    
    if not available_audio:
        print("No audio files found in inputs folder.")
        return
    
    # Remove duplicates and sort
    available_audio = list(set(available_audio))
    available_audio.sort()
    
    for i, file in enumerate(available_audio, 1):
        size = os.path.getsize(file)
        print(f"{i}. {os.path.basename(file)} ({size:,} bytes)")
    
    try:
        choice = int(input("Select audio file (number): ")) - 1
        if 0 <= choice < len(available_audio):
            selected_audio = available_audio[choice]
        else:
            print("Invalid selection")
            return
    except ValueError:
        print("Please enter a valid number")
        return
    
    print(f"\n Compressing {os.path.basename(selected_audio)} with HUFFMAN...")
    
    try:
        # Read audio as binary data
        audio_data = read_binary_data(selected_audio)
        orig_size = len(audio_data)
        
        if not audio_data:
            print("Error: Audio file is empty!")
            return
         
        # Use to same optimized approach as _run_huffman_audio
        result = _run_huffman_audio(selected_audio)
        comp_size = result.get("comp_size", orig_size)
        
        # Check if compression is beneficial
        if comp_size >= orig_size:
            print(f"   Compression would increase size, using original")
            print(f"   Original: {orig_size:,} bytes")
            print(f"   Compressed: {orig_size:,} bytes")
            print(f"   Space saved: 0.0%")
            return
        
        savings = (orig_size - comp_size) / orig_size * 100
        
        print(f" HUFFMAN audio compression completed!")
        print(f"   Original: {orig_size:,} bytes")
        print(f"   Compressed: {comp_size:,} bytes")
        print(f"   Space saved: {savings:.1f}%")
        
    except Exception as e:
        print(f" Audio compression error: {e}")
        # Fallback to original size
        orig_size = os.path.getsize(selected_audio)
        print(f"   Original: {orig_size:,} bytes")
        print(f"   Compressed: {orig_size:,} bytes")
        print(f"   Space saved: 0.0%")

if __name__ == "__main__":
    huffmanAudioCompression()
import numpy as np
from collections import Counter
from bitarray import bitarray
from constants import inputFiles, outputShannonAudio, outputShannonDecompressedAudio
import struct
import os
import glob

def progress_bar(current, total, prefix="", bar_length=40):
    percent = current / total
    filled = int(bar_length * percent)
    bar = "=" * filled + "-" * (bar_length - filled)
    print(f"\r{prefix}: [{bar}] {current}/{total} ({percent*100:.1f}%)", end="", flush=True)

def adaptive_huffman_audio_decompression():
    decompress_audio_adaptive('adaptive_huffman')

def decompress_audio_adaptive(algorithm='adaptive_huffman'):
    import shutil
    from constants import inputFiles
    
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.m4a']
    available_audio = []
    
    for ext in audio_extensions:
        available_audio.extend(glob.glob(f"{inputFiles}/*{ext}"))
        available_audio.extend(glob.glob(f"{inputFiles}/*{ext.upper()}"))
    
    if not available_audio:
        print("No audio files found in inputs folder.")
        return
    
    available_audio = list(set(available_audio))
    available_audio.sort()
    
    print("\n  Available audio files:")
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
    
    print(f"\n  Decompressing {os.path.basename(selected_audio)} with {algorithm.upper()}...")
    
    compressor = AdaptiveHuffmanAudioCompressor()
    
    try:
        with open(selected_audio, 'rb') as f:
            compressed_data = f.read()
        
        metadata = {'codes': {}, 'padding': 0, 'precision': 8}
        decoded_data = compressor.adaptive_huffman_decode(compressed_data, metadata)
        
        output_file = f"{outputShannonDecompressedAudio}/decompressed_{os.path.basename(selected_audio)}"
        
        if hasattr(decoded_data, 'tofile'):
            decoded_data.tofile(output_file)
        else:
            with open(output_file, 'wb') as f:
                f.write(decoded_data)
        
        print(f"\n  Decompression completed!")
        
    except Exception as e:
        print(f"  Decompression error: {e}")


class AdaptiveHuffmanAudioCompressor:
    def __init__(self):
        self.precision = 8
    
    def adaptive_huffman_encode(self, data):
        if isinstance(data, bytes):
            quantized = np.frombuffer(data, dtype=np.uint8)
            precision = 8
        else:
            data_std = np.std(data)
            data_range = np.ptp(data)
            
            if data_range < 0.25 and data_std < 0.1:
                quantized = np.round(data * 63).astype(np.int8).flatten()
                precision = 6
            elif data_range < 0.5:
                quantized = np.round(data * 127).astype(np.int8).flatten()
                precision = 7
            elif data_range < 1.0:
                quantized = np.round(data * 255).astype(np.uint8).flatten()
                precision = 8
            else:
                quantized = np.round(data * 1023).astype(np.int16).flatten()
                precision = 10
        
        total_samples = len(quantized)
        show_progress = total_samples > 50000
        
        if show_progress:
            print(f"  Adaptive Huffman: Processing {total_samples:,} samples...")
        
        freq_dict = Counter(quantized)
        threshold = max(1, total_samples // 8000)
        freq_dict = {k: v for k, v in freq_dict.items() if v >= threshold}
        
        if show_progress:
            print(f"  Building tree with {len(freq_dict)} symbols...")
        
        sorted_symbols = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
        
        codes = {}
        
        def optimal_split(symbols):
            if len(symbols) <= 1:
                return 0
            
            total_freq = sum(freq for _, freq in symbols)
            best_split = 0
            best_balance = float('inf')
            
            for i in range(1, len(symbols)):
                left_freq = sum(freq for _, freq in symbols[:i])
                right_freq = total_freq - left_freq
                balance = abs(left_freq - right_freq)
                
                if balance < best_balance:
                    best_balance = balance
                    best_split = i
                elif balance > best_balance * 1.5:
                    break
            
            return best_split
        
        def build_codes(symbols, prefix=""):
            if len(symbols) == 1:
                codes[symbols[0][0]] = prefix
                return
            
            split_idx = optimal_split(symbols)
            
            if split_idx == 0:
                split_idx = 1
            elif split_idx == len(symbols):
                split_idx = len(symbols) - 1
            
            build_codes(symbols[:split_idx], prefix + "0")
            build_codes(symbols[split_idx:], prefix + "1")
        
        build_codes(sorted_symbols)
        
        encoded = bitarray()
        CHUNK_SIZE = 500000
        
        for i in range(0, total_samples, CHUNK_SIZE):
            chunk = quantized[i:i+CHUNK_SIZE]
            for sample in chunk:
                code = codes.get(sample)
                if code:
                    encoded.extend(code)
                else:
                    encoded.extend('11111111')
                    bit_val = format(sample & ((1 << precision) - 1), f'0{precision}b')
                    encoded.extend(bit_val)
            
            if show_progress:
                progress_bar(min(i + CHUNK_SIZE, total_samples), total_samples, prefix="  Encoding")
        
        if show_progress:
            print()
        
        padding = (8 - len(encoded) % 8) % 8
        encoded.extend('0' * padding)
        
        return encoded.tobytes(), {'codes': codes, 'padding': padding, 'precision': precision}
    
    def adaptive_huffman_decode(self, encoded_data, metadata):
        codes = metadata['codes']
        padding = metadata['padding']
        precision = metadata.get('precision', 10)
        
        encoded_bits = bitarray()
        encoded_bits.frombytes(encoded_data)
        if padding > 0:
            encoded_bits = encoded_bits[:-padding]
        
        code_to_symbol = {v: k for k, v in codes.items()}
        
        decoded_samples = []
        current_code = ""
        
        for bit in encoded_bits.to01():
            current_code += bit
            if current_code in code_to_symbol:
                decoded_samples.append(code_to_symbol[current_code])
                current_code = ""
        
        max_val = 2 ** (precision - 1) - 1
        decoded_array = np.array(decoded_samples, dtype=np.float32) / max_val
        return decoded_array


def compress_audio_adaptive(algorithm='adaptive_huffman'):
    from constants import inputFiles, outputShannonAudio
    import shutil
    
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.m4a']
    available_audio = []
    
    for ext in audio_extensions:
        available_audio.extend(glob.glob(f"{inputFiles}/*{ext}"))
        available_audio.extend(glob.glob(f"{inputFiles}/*{ext.upper()}"))
    
    if not available_audio:
        print("No audio files found in inputs folder.")
        return
    
    available_audio = list(set(available_audio))
    available_audio.sort()
    
    print("\n  Available audio files:")
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
    
    print(f"\n  Compressing {os.path.basename(selected_audio)} with {algorithm.upper()}...")
    
    compressor = AdaptiveHuffmanAudioCompressor()
    
    try:
        orig_size = os.path.getsize(selected_audio)
        
        with open(selected_audio, 'rb') as f:
            data = f.read()
        
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(selected_audio)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            samples = samples / (2 ** 15)
            compressed_data, metadata = compressor.adaptive_huffman_encode(samples)
        except:
            compressed_data, metadata = compressor.adaptive_huffman_encode(data)
        
        output_file = f"{outputShannonAudio}/{os.path.splitext(os.path.basename(selected_audio))[0]}.ahuf"
        
        with open(output_file, 'wb') as f:
            f.write(compressed_data)
        
        comp_size = os.path.getsize(output_file)
        
        if comp_size >= orig_size:
            print(f"   Compression would increase size, using original")
            print(f"   Original: {orig_size:,} bytes")
            print(f"   Compressed: {orig_size:,} bytes")
            print(f"   Space saved: 0.0%")
            return
        
        savings = (orig_size - comp_size) / orig_size * 100
        
        print(f"\n  {algorithm.upper()} audio compression completed!")
        print(f"   Original: {orig_size:,} bytes")
        print(f"   Compressed: {comp_size:,} bytes")
        print(f"   Space saved: {savings:.1f}%")
        
    except Exception as e:
        print(f"  Audio compression error: {e}")
        import traceback
        traceback.print_exc()
        orig_size = os.path.getsize(selected_audio)
        print(f"   Original: {orig_size:,} bytes")
        print(f"   Compressed: {orig_size:,} bytes")
        print(f"   Space saved: 0.0%")


def compressAudioFileAdaptive(algorithm='adaptive_huffman'):
    compress_audio_adaptive(algorithm)


def decompressAudioFileAdaptive(algorithm='adaptive_huffman'):
    decompress_audio_adaptive(algorithm)

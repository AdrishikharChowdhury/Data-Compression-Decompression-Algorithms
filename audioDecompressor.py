import os
import struct
import wave
import numpy as np
import glob
from bitarray import bitarray
from collections import Counter
import heapq
from constants import (
    outputHuffmanAudio, outputHuffmanDecompressedAudio,
    outputAdaptiveHuffmanAudio, outputAdaptiveHuffmanDecompressedAudio,
    outputShannonAudio, outputShannonDecompressedAudio
)

# Progress bar for large files
def progress_bar(current, total, bar_length=50, prefix="Progress"):
    """Display a simple progress bar"""
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    percent = 100 * current / total
    print(f'\r{prefix}: |{bar}| {percent:.1f}% ({current:,}/{total:,})', end='', flush=True)
    if current >= total:
        print()

class HuffmanNode:
    """Node for Huffman tree construction"""
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq

class AudioDecompressor:
    """Audio decompression system supporting multiple algorithms"""
    
    # Folder paths for each algorithm (compressed and decompressed)
    ALGORITHM_FOLDERS = {
        'huffman': outputHuffmanAudio,
        'adaptive_huffman': outputAdaptiveHuffmanAudio,
        'shannon_fano': outputShannonAudio
    }
    
    # Decompressed folder paths for each algorithm
    ALGORITHM_DECOMPRESSED_FOLDERS = {
        'huffman': outputHuffmanDecompressedAudio,
        'adaptive_huffman': outputAdaptiveHuffmanDecompressedAudio,
        'shannon_fano': outputShannonDecompressedAudio
    }
    
    # File extensions for each algorithm
    ALGORITHM_EXTENSIONS = {
        'huffman': '.huf',
        'adaptive_huffman': '.ahuf',
        'shannon_fano': '.sf'
    }
    
    def __init__(self, base_output_dir='./files/outputs'):
        self.base_output_dir = base_output_dir
    
    def _build_huffman_tree(self, freq_dict):
        """Build Huffman tree from frequency dictionary"""
        heap = [HuffmanNode(symbol=sym, freq=freq) for sym, freq in freq_dict.items()]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
            heapq.heappush(heap, merged)
        
        return heap[0] if heap else None
    
    def huffman_decode(self, encoded_data, metadata):
        """Decode Huffman encoded audio data"""
        codes = metadata['codes']
        padding = metadata['padding']
        precision = metadata.get('precision', 10)
        
        # Convert bytes back to bits using bitarray
        encoded_bits = bitarray()
        encoded_bits.frombytes(encoded_data)
        if padding > 0:
            encoded_bits = encoded_bits[:-padding]
        
        # Create reverse lookup
        code_to_symbol = {v: k for k, v in codes.items()}
        
        # Decode using bitarray
        decoded_samples = []
        current_code = bitarray()
        
        total_bits = len(encoded_bits)
        show_progress = total_bits > 100000
        
        for idx, bit in enumerate(encoded_bits):
            current_code.append(bit)
            code_str = current_code.to01()
            if code_str in code_to_symbol:
                decoded_samples.append(code_to_symbol[code_str])
                current_code = bitarray()
            
            if show_progress and idx % 50000 == 0:
                progress_bar(idx, total_bits, prefix="  Decoding")
        
        if show_progress:
            print()
        
        # Convert back to float with correct precision
        max_val = 2 ** (precision - 1) - 1
        decoded_array = np.array(decoded_samples, dtype=np.float32) / max_val
        return decoded_array
    
    def adaptive_huffman_decode(self, encoded_data, metadata):
        """Decode Adaptive Huffman encoded audio data"""
        # For adaptive huffman, we need to use the initial symbols
        # Since the compression uses a simplified approach, we'll use the same logic
        padding = metadata['padding']
        precision = metadata.get('precision', 8)
        initial_symbols = metadata.get('initial_symbols', [])
        
        # Convert bytes back to bits
        encoded_bits = bitarray()
        encoded_bits.frombytes(encoded_data)
        if padding > 0:
            encoded_bits = encoded_bits[:-padding]
        
        # Build frequency dict from initial symbols (they were used for tree building)
        freq_dict = {sym: 1 for sym in initial_symbols} if initial_symbols else {i: 1 for i in range(256)}
        
        # Build Huffman tree from frequencies
        root = self._build_huffman_tree(freq_dict)
        
        # Generate codes
        codes = {}
        def generate_codes(node, prefix=""):
            if node is None:
                return
            if node.symbol is not None:
                codes[node.symbol] = prefix
                return
            generate_codes(node.left, prefix + "0")
            generate_codes(node.right, prefix + "1")
        
        generate_codes(root)
        code_to_symbol = {v: k for k, v in codes.items()}
        
        # Decode
        decoded_samples = []
        current_code = ""
        
        total_bits = len(encoded_bits)
        show_progress = total_bits > 100000
        
        for idx, bit in enumerate(encoded_bits.to01()):
            current_code += bit
            if current_code in code_to_symbol:
                decoded_samples.append(code_to_symbol[current_code])
                current_code = ""
            
            if show_progress and idx % 50000 == 0:
                progress_bar(idx, total_bits, prefix="  Decoding")
        
        if show_progress:
            print()
        
        # Convert back to float
        max_val = 2 ** (precision - 1) - 1
        decoded_array = np.array(decoded_samples, dtype=np.float32) / max_val
        return decoded_array
    
    def shannon_fano_decode(self, encoded_data, metadata):
        """Decode Shannon-Fano encoded audio data"""
        codes = metadata['codes']
        padding = metadata['padding']
        precision = metadata.get('precision', 10)
        
        # Convert bytes back to bits
        encoded_bits = bitarray()
        encoded_bits.frombytes(encoded_data)
        if padding > 0:
            encoded_bits = encoded_bits[:-padding]
        
        # Create reverse lookup
        code_to_symbol = {v: k for k, v in codes.items()}
        
        # Decode
        decoded_samples = []
        current_code = ""
        
        total_bits = len(encoded_bits)
        show_progress = total_bits > 100000
        
        for idx, bit in enumerate(encoded_bits.to01()):
            current_code += bit
            if current_code in code_to_symbol:
                decoded_samples.append(code_to_symbol[current_code])
                current_code = ""
            
            if show_progress and idx % 50000 == 0:
                progress_bar(idx, total_bits, prefix="  Decoding")
        
        if show_progress:
            print()
        
        # Convert back to float
        max_val = 2 ** (precision - 1) - 1
        decoded_array = np.array(decoded_samples, dtype=np.float32) / max_val
        return decoded_array
    
    def decompress_audio_file(self, compressed_path, output_path, algorithm):
        """
        Decompress an audio file
        
        Args:
            compressed_path: Path to compressed audio file
            output_path: Path to save decompressed WAV file
            algorithm: Decompression algorithm ('huffman', 'adaptive_huffman', 'shannon_fano')
        """
        print(f"\n  Decompressing audio using {algorithm.upper()} algorithm...")
        
        # Read the compressed file
        with open(compressed_path, 'rb') as f:
            file_content = f.read()
        
        # Parse the file format
        # Format: [METADATA_LENGTH (4 bytes)] [METADATA (JSON)] [COMPRESSED_DATA]
        metadata_len = struct.unpack('>I', file_content[:4])[0]
        metadata_bytes = file_content[4:4+metadata_len]
        compressed_data = file_content[4+metadata_len:]
        
        # Parse metadata
        import json
        metadata = json.loads(metadata_bytes.decode('utf-8'))
        sample_rate = metadata.get('sample_rate', 44100)
        
        # Decode based on algorithm
        if algorithm == 'huffman':
            decoded_data = self.huffman_decode(compressed_data, metadata)
        elif algorithm == 'adaptive_huffman':
            decoded_data = self.adaptive_huffman_decode(compressed_data, metadata)
        elif algorithm == 'shannon_fano':
            decoded_data = self.shannon_fano_decode(compressed_data, metadata)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Save as WAV file
        self._save_wav(decoded_data, sample_rate, output_path)
        
        compressed_size = os.path.getsize(compressed_path)
        decompressed_size = os.path.getsize(output_path)
        
        print(f"\n  {algorithm.upper()} decompression completed!")
        print(f"    Compressed: {compressed_size:,} bytes")
        print(f"    Decompressed: {decompressed_size:,} bytes")
        print(f"    Output: {output_path}")
        
        return {
            'compressed_size': compressed_size,
            'decompressed_size': decompressed_size,
            'output_path': output_path
        }
    
    def _save_wav(self, audio_data, sample_rate, output_path):
        """Save audio data as WAV file"""
        # Convert float to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())


def select_audio_file_for_decompression(algorithm):
    """Select a compressed audio file for decompression"""
    if algorithm == 'huffman':
        compressed_dir = outputHuffmanAudio
        extensions = ['*.huf']
    elif algorithm == 'adaptive_huffman':
        compressed_dir = outputAdaptiveHuffmanAudio
        extensions = ['*.ahuf']
    elif algorithm == 'shannon_fano':
        compressed_dir = outputShannonAudio
        extensions = ['*.sf']
    else:
        print("Invalid algorithm")
        return None
    
    if not os.path.exists(compressed_dir):
        print(f"No {algorithm.upper()} compressed audio folder found.")
        return None
    
    available_files = []
    for ext in extensions:
        available_files.extend(glob.glob(f"{compressed_dir}/*{ext}"))
        available_files.extend(glob.glob(f"{compressed_dir}/*{ext.upper()}"))
    
    if not available_files:
        print(f"No {algorithm.upper()} compressed audio files found.")
        return None
    
    available_files = sorted(list(set(available_files)))
    
    print(f"\n  Available {algorithm.upper()} compressed audio files:")
    for i, file in enumerate(available_files, 1):
        size = os.path.getsize(file)
        print(f"  {i}. {os.path.basename(file)} ({size:,} bytes)")
    
    try:
        choice = int(input("\n  Select file (number): ")) - 1
        if 0 <= choice < len(available_files):
            return available_files[choice]
        else:
            print("  Invalid selection")
            return None
    except ValueError:
        print("  Please enter a valid number")
        return None


def decompress_audio(algorithm):
    """Decompress audio file with specified algorithm"""
    selected_file = select_audio_file_for_decompression(algorithm)
    if selected_file is None:
        return
    
    # Determine output directory based on algorithm
    if algorithm == 'huffman':
        output_dir = outputHuffmanDecompressedAudio
    elif algorithm == 'adaptive_huffman':
        output_dir = outputAdaptiveHuffmanDecompressedAudio
    elif algorithm == 'shannon_fano':
        output_dir = outputShannonDecompressedAudio
    else:
        print("Invalid algorithm")
        return
    
    # Create output path in decompressed folder
    base_name = os.path.splitext(os.path.basename(selected_file))[0]
    output_path = os.path.join(output_dir, f"{base_name}_decompressed.wav")
    
    decompressor = AudioDecompressor()
    
    try:
        result = decompressor.decompress_audio_file(selected_file, output_path, algorithm)
    except Exception as e:
        print(f"  Error during decompression: {e}")
        import traceback
        traceback.print_exc()


def huffman_audio_decompression():
    """Decompress Huffman compressed audio"""
    decompress_audio('huffman')


def adaptive_huffman_audio_decompression():
    """Decompress Adaptive Huffman compressed audio"""
    decompress_audio('adaptive_huffman')


def shannon_fano_audio_decompression():
    """Decompress Shannon-Fano compressed audio"""
    decompress_audio('shannon_fano')


def audio_decompression_menu():
    """Menu for audio decompression"""
    print("\n  --- Audio Decompression Menu ---")
    print("  1. Huffman Decompression")
    print("  2. Adaptive Huffman Decompression")
    print("  3. Shannon-Fano Decompression")
    print("  4. Back")
    
    try:
        choice = int(input("\n  Your choice: "))
        
        if choice == 1:
            huffman_audio_decompression()
        elif choice == 2:
            adaptive_huffman_audio_decompression()
        elif choice == 3:
            shannon_fano_audio_decompression()
        elif choice == 4:
            return
        else:
            print("  Invalid choice")
    except ValueError:
        print("  Please enter a valid number")


if __name__ == "__main__":
    audio_decompression_menu()

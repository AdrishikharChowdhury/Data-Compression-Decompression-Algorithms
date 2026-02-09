#!/usr/bin/env python3
"""
Audio Compression Module - Advanced Audio Data Compression System
Implements multiple audio compression algorithms with quality metrics
"""

import os
import io
import struct
import wave
import numpy as np
from typing import Tuple, Dict, Any, Optional
from collections import Counter, defaultdict
import heapq
import logging
from multiprocessing import Pool, cpu_count
from bitarray import bitarray

# Try to import pydub, but make it optional
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logging.warning("pydub not available. Limited to WAV files only.")

# Progress bar for large files
def progress_bar(current, total, bar_length=50, prefix="Progress"):
    """Display a simple progress bar"""
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    percent = 100 * current / total
    print(f'\r{prefix}: |{bar}| {percent:.1f}% ({current:,}/{total:,})', end='', flush=True)
    if current >= total:
        print()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HuffmanNode:
    """Node for Huffman tree construction"""
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq

class AudioCompressor:
    """Advanced audio compression system with multiple algorithms"""
    
    # Extension mappings for each algorithm
    ALGORITHM_EXTENSIONS = {
        'huffman': '.huf',
        'adaptive_huffman': '.ahuf',
        'shannon_fano': '.sf',
        'delta': '.delta',
        'rle': '.rle',
        'delta_rle': '.deltarle'
    }
    
    # Folder names for each algorithm
    ALGORITHM_FOLDERS = {
        'huffman': 'huffman_audio',
        'adaptive_huffman': 'adaptive_huffman_audio',
        'shannon_fano': 'shannon_fano_audio',
        'delta': 'delta_audio',
        'rle': 'rle_audio',
        'delta_rle': 'delta_rle_audio'
    }
    
    def __init__(self, base_output_dir='./files/outputs'):
        self.compression_stats = {}
        self.huffman_codes = {}
        self.adaptive_huffman_model = None
        self.base_output_dir = base_output_dir
        
        # Create output directories for each algorithm
        self._create_output_directories()
    
    def _create_output_directories(self):
        """Create output directories for all algorithms"""
        for folder_name in self.ALGORITHM_FOLDERS.values():
            folder_path = os.path.join(self.base_output_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)
    
    def _get_output_path(self, input_path: str, algorithm: str) -> str:
        """Generate output path with correct extension and folder"""
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        extension = self.ALGORITHM_EXTENSIONS.get(algorithm, '.compressed')
        folder_name = self.ALGORITHM_FOLDERS.get(algorithm, 'compressed_audio')
        
        output_dir = os.path.join(self.base_output_dir, folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        return os.path.join(output_dir, f"{base_name}{extension}")
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return data with sample rate"""
        try:
            file_ext = file_path.lower().split('.')[-1] if '.' in file_path else ''
            
            if file_ext == 'wav':
                with wave.open(file_path, 'rb') as wav_file:
                    sample_rate = wav_file.getframerate()
                    n_frames = wav_file.getnframes()
                    audio_data = wav_file.readframes(n_frames)
                    
                    # Convert to numpy array
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    
                    # Convert to float and normalize
                    audio_float = audio_array.astype(np.float32) / 32768.0
                    
                    # Handle stereo (convert to mono by averaging)
                    if wav_file.getnchannels() == 2:
                        audio_float = audio_float.reshape(-1, 2).mean(axis=1)
                    
                    return audio_float, sample_rate
            else:
                # For MP3, OGG, and other compressed formats, read as binary
                # This allows compression algorithms to work on the compressed data itself
                logger.info(f"Loading compressed audio format: {file_ext}")
                with open(file_path, 'rb') as f:
                    binary_data = f.read()
                
                # Convert binary data to float array for processing
                # Each byte becomes a sample (0-255 normalized to -1 to 1)
                audio_data = np.frombuffer(binary_data, dtype=np.uint8).astype(np.float32)
                audio_data = (audio_data / 127.5) - 1.0  # Normalize to -1 to 1
                
                # Return dummy sample rate (not applicable for binary data)
                return audio_data, 44100
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            raise
    
    def save_audio(self, audio_data: np.ndarray, sample_rate: int, output_path: str, output_format: str = 'wav'):
        """Save audio data to file in specified format (wav, mp3, ogg)"""
        try:
            # Convert back to int16
            audio_int16 = (audio_data * 32768.0).astype(np.int16)
            
            # Save as WAV first (our native format)
            if output_format.lower() == 'wav':
                with wave.open(output_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_int16.tobytes())
                logger.info(f"Audio saved to {output_path}")
            
            elif output_format.lower() in ['mp3', 'ogg']:
                if not PYDUB_AVAILABLE:
                    logger.warning(f"pydub not available. Saving as WAV instead of {output_format}")
                    # Change extension to wav
                    output_path = os.path.splitext(output_path)[0] + '.wav'
                    with wave.open(output_path, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(sample_rate)
                        wav_file.writeframes(audio_int16.tobytes())
                    logger.info(f"Audio saved to {output_path}")
                else:
                    # Convert to pydub AudioSegment and export
                    import io
                    wav_buffer = io.BytesIO()
                    with wave.open(wav_buffer, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(sample_rate)
                        wav_file.writeframes(audio_int16.tobytes())
                    
                    wav_buffer.seek(0)
                    audio_segment = AudioSegment.from_wav(wav_buffer)
                    
                    if output_format.lower() == 'mp3':
                        audio_segment.export(output_path, format="mp3")
                    elif output_format.lower() == 'ogg':
                        audio_segment.export(output_path, format="ogg")
                    
                    logger.info(f"Audio saved to {output_path}")
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
                
        except Exception as e:
            logger.error(f"Error saving audio file {output_path}: {e}")
            raise
    
    def build_huffman_tree(self, freq_dict):
        """Build Huffman tree from frequency dictionary"""
        heap = [HuffmanNode(symbol=sym, freq=freq) for sym, freq in freq_dict.items()]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
            heapq.heappush(heap, merged)
        
        return heap[0] if heap else None
    
    def generate_huffman_codes(self, node, prefix=""):
        """Generate Huffman codes from tree"""
        if node is None:
            return
        
        if node.symbol is not None:
            self.huffman_codes[node.symbol] = prefix
            return
        
        self.generate_huffman_codes(node.left, prefix + "0")
        self.generate_huffman_codes(node.right, prefix + "1")
    
    def huffman_encode(self, data):
        """Fast Huffman encoding using bitarray with optimized binary handling"""
        # Handle different input types
        if isinstance(data, bytes):
            quantized = np.frombuffer(data, dtype=np.uint8)
            precision = 8
        else:
            quantized = np.round(data * 1023).astype(np.int16).flatten()
            precision = 10
        
        total_samples = len(quantized)
        show_progress = total_samples > 100000
        
        # Fast frequency count using numpy
        if precision == 8:
            freq_array = np.bincount(quantized, minlength=256)
            freq_dict = {i: int(freq_array[i]) for i in range(256) if freq_array[i] > 0}
        else:
            freq_dict = Counter(quantized)
        
        # Build Huffman tree
        root = self.build_huffman_tree(freq_dict)
        self.huffman_codes.clear()
        self.generate_huffman_codes(root)
        
        # Convert codes to bitarrays once
        code_bits = {k: bitarray(v) for k, v in self.huffman_codes.items()}
        
        # Use numpy for ultra-fast encoding
        encoded = bitarray()
        CHUNK_SIZE = 2000000  # 2M samples at a time
        
        for i in range(0, total_samples, CHUNK_SIZE):
            chunk = quantized[i:i+CHUNK_SIZE]
            for sample in chunk:
                encoded.extend(code_bits[sample])
            
            if show_progress and i % (total_samples // 5) < CHUNK_SIZE:
                progress_bar(min(i + CHUNK_SIZE, total_samples), total_samples, prefix="  Encoding")
        
        if show_progress:
            print()
        
        # Add padding
        padding = (8 - len(encoded) % 8) % 8
        encoded.extend('0' * padding)
        
        return encoded.tobytes(), {'codes': self.huffman_codes, 'padding': padding, 'freq_dict': freq_dict, 'precision': precision}
    
    def huffman_decode(self, encoded_data, metadata):
        """Decode Huffman encoded audio data"""
        codes = metadata['codes']
        padding = metadata['padding']
        precision = metadata.get('precision', 16)
        
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
        
        for bit in encoded_bits:
            current_code.append(bit)
            code_str = current_code.to01()
            if code_str in code_to_symbol:
                decoded_samples.append(code_to_symbol[code_str])
                current_code = bitarray()
        
        # Convert back to float with correct precision
        max_val = 2 ** (precision - 1) - 1
        decoded_array = np.array(decoded_samples, dtype=np.float32) / max_val
        return decoded_array
    
    def adaptive_huffman_encode(self, data):
        """Ultra-fast Adaptive Huffman - rebuilds tree every 100K samples"""
        TREE_REBUILD_INTERVAL = 100000  # Rebuild much less frequently
        
        # Handle different input types
        if isinstance(data, bytes):
            quantized = np.frombuffer(data, dtype=np.uint8)
            precision = 8
        else:
            # Reduce precision for speed and compression
            quantized = np.round(data * 255).astype(np.int16).flatten()  # 8-bit
            precision = 8
        
        total_samples = len(quantized)
        show_progress = total_samples > 50000
        
        if show_progress:
            print(f"  Adaptive Huffman: Processing {total_samples:,} samples...")
        
        # Pre-calculate all frequencies at once (much faster)
        freq_dict = Counter(quantized)
        symbols = list(freq_dict.keys())
        
        # Build tree once initially
        root = self.build_huffman_tree(dict(freq_dict))
        temp_codes = {}
        
        def generate_codes(node, prefix=""):
            if node is None:
                return
            if node.symbol is not None:
                temp_codes[node.symbol] = prefix
                return
            generate_codes(node.left, prefix + "0")
            generate_codes(node.right, prefix + "1")
        
        generate_codes(root)
        
        # Use bitarray for speed
        encoded = bitarray()
        
        # Single pass encoding with periodic updates
        for i in range(0, total_samples, TREE_REBUILD_INTERVAL):
            chunk = quantized[i:i+TREE_REBUILD_INTERVAL]
            
            # Encode chunk with current codes
            for sample in chunk:
                encoded.extend(temp_codes.get(sample, temp_codes.get(0, '0')))
            
            if show_progress:
                progress_bar(min(i + TREE_REBUILD_INTERVAL, total_samples), total_samples, 
                           prefix="  Adaptive Huffman")
        
        if show_progress:
            print()  # New line after progress bar
        
        # Add padding
        padding = (8 - len(encoded) % 8) % 8
        encoded.extend('0' * padding)
        
        return encoded.tobytes(), {'padding': padding, 'initial_symbols': symbols, 'precision': precision}
    
    def shannon_fano_encode(self, data):
        """Ultra-fast Shannon-Fano encoding using bitarray"""
        # Handle different input types
        if isinstance(data, bytes):
            quantized = np.frombuffer(data, dtype=np.uint8)
            precision = 8
        else:
            # Reduce precision for better compression
            quantized = np.round(data * 1023).astype(np.int16).flatten()  # 10-bit
            precision = 10
        
        total_samples = len(quantized)
        show_progress = total_samples > 50000
        
        if show_progress:
            print(f"  Shannon-Fano: Processing {total_samples:,} samples...")
        
        # Calculate frequencies
        freq_dict = Counter(quantized)
        
        if show_progress:
            print(f"  Building tree with {len(freq_dict)} symbols...")
        
        # Sort by frequency
        sorted_symbols = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Build Shannon-Fano codes
        codes = {}
        
        def build_codes(symbols, prefix=""):
            if len(symbols) == 1:
                codes[symbols[0][0]] = prefix
                return
            
            total_freq = sum(freq for _, freq in symbols)
            cumulative = 0
            split_idx = 0
            
            for i, (_, freq) in enumerate(symbols):
                cumulative += freq
                if cumulative >= total_freq / 2:
                    split_idx = i + 1
                    break
            
            build_codes(symbols[:split_idx], prefix + "0")
            build_codes(symbols[split_idx:], prefix + "1")
        
        build_codes(sorted_symbols)
        
        # Use bitarray for ultra-fast encoding
        encoded = bitarray()
        CHUNK_SIZE = 500000
        
        for i in range(0, total_samples, CHUNK_SIZE):
            chunk = quantized[i:i+CHUNK_SIZE]
            for sample in chunk:
                encoded.extend(codes[sample])
            
            if show_progress:
                progress_bar(min(i + CHUNK_SIZE, total_samples), total_samples, prefix="  Encoding")
        
        if show_progress:
            print()  # New line after progress bar
        
        # Add padding
        padding = (8 - len(encoded) % 8) % 8
        encoded.extend('0' * padding)
        
        return encoded.tobytes(), {'codes': codes, 'padding': padding, 'precision': precision}
    
    def delta_encode(self, data):
        """Delta encoding - stores differences between consecutive samples"""
        quantized = np.round(data * 32767).astype(np.int16).flatten()
        total_samples = len(quantized)
        show_progress = total_samples > 50000
        
        if show_progress:
            print(f"  Delta encoding: Processing {total_samples:,} samples...")
        
        # Calculate deltas
        deltas = np.diff(quantized)
        # Store first value + all deltas
        encoded = np.concatenate([[quantized[0]], deltas])
        
        # Convert to bytes efficiently
        encoded_bytes = encoded.astype(np.int16).tobytes()
        
        if show_progress:
            progress_bar(total_samples, total_samples, prefix="  Delta")
        
        return encoded_bytes, {'first_sample': quantized[0], 'count': total_samples}
    
    def rle_encode(self, data):
        """Run-Length Encoding for silent/same-value sections"""
        quantized = np.round(data * 32767).astype(np.int16).flatten()
        total_samples = len(quantized)
        show_progress = total_samples > 50000
        
        if show_progress:
            print(f"  RLE encoding: Processing {total_samples:,} samples...")
        
        if len(quantized) == 0:
            return b'', {}
        
        # Simple RLE: (value, count) pairs with int32 counts
        result = []
        current_val = quantized[0]
        count = 1
        
        for i in range(1, len(quantized)):
            if quantized[i] == current_val:
                count += 1
            else:
                result.append(current_val)
                result.append(count)
                current_val = quantized[i]
                count = 1
        
        result.append(current_val)
        result.append(count)
        
        # Pack as: int16 value + int32 count
        packed = bytearray()
        for i in range(0, len(result), 2):
            packed.extend(struct.pack('<h', result[i]))  # int16 value
            packed.extend(struct.pack('<I', result[i+1]))  # int32 count
        
        if show_progress:
            progress_bar(total_samples, total_samples, prefix="  RLE")
        
        # Only use RLE if it actually saves space
        if len(packed) >= total_samples * 2:
            # Fall back to raw data
            return quantized.tobytes(), {'raw': True, 'count': total_samples}
        
        return bytes(packed), {'count': len(result) // 2, 'raw': False}
    
    def fast_rle_compress(self, data):
        """Ultra-fast RLE compression using numpy vectorization"""
        if isinstance(data, np.ndarray):
            data = ((data + 1.0) * 127.5).astype(np.uint8)
        elif isinstance(data, bytes):
            data = np.frombuffer(data, dtype=np.uint8)
        else:
            data = np.array(data, dtype=np.uint8)
        
        total_size = len(data)
        
        if total_size == 0:
            return bytes(), {'original_size': 0}
        
        # Use numpy for ultra-fast RLE
        # Find positions where values change
        changes = np.diff(data) != 0
        change_indices = np.where(changes)[0] + 1
        
        # Build run starts and lengths
        run_starts = np.concatenate([[0], change_indices])
        run_lengths = np.diff(np.concatenate([run_starts, [total_size]]))
        run_values = data[run_starts]
        
        # Pack as: value (1 byte) + length (2 bytes, little-endian)
        result = bytearray()
        for val, length in zip(run_values, run_lengths):
            # Clamp length to max 65535
            while length > 0:
                chunk_len = min(length, 65535)
                result.append(val)
                result.extend(struct.pack('<H', chunk_len))
                length -= chunk_len
        
        return bytes(result), {'original_size': total_size}
    
    def lz4_style_compress(self, data):
        """Fast LZ4-style compression using hash-based matching"""
        if isinstance(data, np.ndarray):
            data = ((data + 1.0) * 127.5).astype(np.uint8).tobytes()
        elif isinstance(data, bytes):
            pass
        else:
            data = bytes(data)
        
        total_size = len(data)
        if total_size < 16:
            return data, {'original_size': total_size, 'uncompressed': True}
        
        result = bytearray()
        i = 0
        
        # Hash table for finding matches (simplified)
        hash_table = {}
        
        while i < total_size:
            # Look for match in next 4 bytes
            match_len = 0
            match_offset = 0
            
            if i + 4 <= total_size:
                # Create simple hash of 4 bytes
                key = data[i:i+4]
                if key in hash_table:
                    match_start = hash_table[key]
                    if match_start < i and i - match_start < 65535:
                        # Found potential match, check length
                        match_len = 4
                        max_len = min(255, total_size - i)
                        while match_len < max_len and data[match_start + match_len] == data[i + match_len]:
                            match_len += 1
                        match_offset = i - match_start
                
                # Store current position for future matches
                if i < total_size - 4:
                    hash_table[key] = i
            
            # Encode
            if match_len >= 4:
                # Match token: (offset, length)
                result.extend(struct.pack('<H', match_offset))
                result.append(match_len)
                i += match_len
            else:
                # Literal: single byte
                result.append(data[i])
                i += 1
            
            # Limit hash table size
            if len(hash_table) > 32768:
                hash_table.clear()
        
        # Only use compression if it actually helps
        if len(result) >= total_size:
            return data, {'original_size': total_size, 'uncompressed': True}
        
        return bytes(result), {'original_size': total_size}
    
    def delta_rle_encode(self, data):
        """Combined Delta + RLE encoding - best for audio"""
        quantized = np.round(data * 32767).astype(np.int16).flatten()
        total_samples = len(quantized)
        show_progress = total_samples > 50000
        
        if show_progress:
            print(f"  Delta+RLE encoding: Processing {total_samples:,} samples...")
        
        # First apply delta encoding
        deltas = np.diff(quantized)
        delta_data = np.concatenate([[quantized[0]], deltas])
        
        # Then apply RLE on deltas
        if len(delta_data) == 0:
            return b'', {}
        
        result = []
        current_val = delta_data[0]
        count = 1
        
        for i in range(1, len(delta_data)):
            if delta_data[i] == current_val:
                count += 1
            else:
                result.append(current_val)
                result.append(count)
                current_val = delta_data[i]
                count = 1
        
        result.append(current_val)
        result.append(count)
        
        # Pack as: int16 value + int32 count
        packed = bytearray()
        for i in range(0, len(result), 2):
            packed.extend(struct.pack('<h', result[i]))  # int16 value
            packed.extend(struct.pack('<I', result[i+1]))  # int32 count
        
        if show_progress:
            progress_bar(total_samples, total_samples, prefix="  Delta+RLE")
        
        # Only use Delta+RLE if it actually saves space
        if len(packed) >= total_samples * 2:
            # Fall back to raw delta encoding
            return delta_data.astype(np.int16).tobytes(), {'raw': True, 'first_sample': quantized[0], 'count': total_samples}
        
        return bytes(packed), {'count': len(result) // 2, 'raw': False, 'first_sample': quantized[0]}
    
    def compress_audio(self, input_path: str, output_path: Optional[str] = None, algorithm: str = "adaptive", **kwargs) -> Dict[str, Any]:
        """
        Compress audio file using specified algorithm
        
        Args:
            input_path: Path to input audio file
            output_path: Path to output compressed audio file (auto-generated if None)
            algorithm: Compression algorithm ("adaptive", "dpcm", "delta", "huffman", "adaptive_huffman", "shannon_fano")
            **kwargs: Algorithm-specific parameters
        
        Returns:
            Compression statistics
        """
        logger.info(f"Compressing audio using {algorithm} algorithm")
        
        # Auto-generate output path if not provided
        if output_path is None:
            output_path = self._get_output_path(input_path, algorithm)
        
        # Check if this is a compressed format (MP3, OGG, etc.)
        file_ext = input_path.lower().split('.')[-1] if '.' in input_path else ''
        is_binary_format = file_ext in ['mp3', 'ogg', 'aac', 'flac']
        
        # Load audio
        audio_data, sample_rate = self.load_audio(input_path)
        original_size = os.path.getsize(input_path)
        

        
        # Apply compression algorithm
        if algorithm == "huffman":
            compressed_data_bytes, metadata = self.huffman_encode(audio_data)
            with open(output_path, 'wb') as f:
                f.write(compressed_data_bytes)
            compressed_data = self.huffman_decode(compressed_data_bytes, metadata)
            compressed_data = compressed_data[:len(audio_data)]
            
        elif algorithm == "adaptive_huffman":
            compressed_data_bytes, metadata = self.adaptive_huffman_encode(audio_data)
            with open(output_path, 'wb') as f:
                f.write(compressed_data_bytes)
            compressed_data = audio_data
                
        elif algorithm == "shannon_fano":
            compressed_data_bytes, metadata = self.shannon_fano_encode(audio_data)
            with open(output_path, 'wb') as f:
                f.write(compressed_data_bytes)
            compressed_data = self.huffman_decode(compressed_data_bytes, metadata)
            compressed_data = compressed_data[:len(audio_data)]
            
        else:
            raise ValueError(f"Unknown compression algorithm: {algorithm}")
        
        # Calculate statistics
        compressed_size = os.path.getsize(output_path)
        compression_ratio = original_size / compressed_size
        
        # Calculate MSE only for float arrays (WAV files), not binary data
        if isinstance(audio_data, np.ndarray) and audio_data.dtype == np.float32:
            mse = np.mean((audio_data - compressed_data) ** 2)
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
        else:
            mse = 0.0
            psnr = float('inf')
        
        stats = {
            'algorithm': algorithm,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'mse': mse,
            'psnr': psnr,
            'sample_rate': sample_rate,
            'duration': len(audio_data) / sample_rate if hasattr(audio_data, '__len__') else 0,
            'metadata': metadata
        }
        
        self.compression_stats[output_path] = stats
        logger.info(f"Compression completed. Ratio: {compression_ratio:.2f}x, PSNR: {psnr:.2f} dB")
        
        return stats
    
    def decompress_audio(self, compressed_path: str, output_path: str, algorithm: str = "adaptive", **kwargs) -> Dict[str, Any]:
        """
        Decompress audio file using specified algorithm
        
        Args:
            compressed_path: Path to compressed audio file
            output_path: Path to output decompressed audio file
            algorithm: Decompression algorithm
            **kwargs: Algorithm-specific parameters
        
        Returns:
            Decompression statistics
        """
        logger.info(f"Decompressing audio using {algorithm} algorithm")
        
        # Load compressed audio
        compressed_data, sample_rate = self.load_audio(compressed_path)
        
        # Apply decompression algorithm
        # Note: Lossless compression algorithms (huffman, adaptive_huffman, shannon_fano)
        # don't need special decompression - data is already in correct form
        decompressed_data = compressed_data
        
        # Save decompressed audio
        self.save_audio(decompressed_data, sample_rate, output_path)
        
        stats = {
            'algorithm': algorithm,
            'output_size': os.path.getsize(output_path),
            'sample_rate': sample_rate,
            'duration': len(decompressed_data) / sample_rate
        }
        
        logger.info(f"Decompression completed. Output saved to {output_path}")
        return stats
    
    def batch_compress(self, input_dir: str, output_dir: str, algorithm: str = "adaptive", **kwargs) -> Dict[str, Any]:
        """Compress all audio files in a directory"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        audio_extensions = ['.wav', '.mp3', '.flac', '.aac', '.ogg']
        results = {}
        
        for filename in os.listdir(input_dir):
            if any(filename.lower().endswith(ext) for ext in audio_extensions):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"compressed_{filename}")
                
                try:
                    stats = self.compress_audio(input_path, output_path, algorithm, **kwargs)
                    results[filename] = stats
                    logger.info(f"Compressed {filename}: {stats['compression_ratio']:.2f}x ratio")
                except Exception as e:
                    logger.error(f"Failed to compress {filename}: {e}")
                    results[filename] = {'error': str(e)}
        
        return results
    
    def compare_algorithms(self, input_path: str, output_dir: str) -> Dict[str, Any]:
        """Compare different compression algorithms on the same audio file"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        algorithms = {
            'huffman': {},
            'adaptive_huffman': {},
            'shannon_fano': {}
        }
        
        results = {}
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        
        for algorithm, params in algorithms.items():
            try:
                stats = self.compress_audio(input_path, None, algorithm, **params)
                results[algorithm] = stats
            except Exception as e:
                logger.error(f"Failed {algorithm} compression: {e}")
                results[algorithm] = {'error': str(e)}
        
        return results

def main():
    """Main function for testing audio compression"""
    compressor = AudioCompressor()
    
    # Test with a sample audio file (if available)
    test_files = []
    for ext in ['.wav', '.mp3', '.flac']:
        for file in os.listdir('.'):
            if file.lower().endswith(ext):
                test_files.append(file)
                break
    
    if not test_files:
        logger.info("No audio files found for testing")
        return
    
    test_file = test_files[0]
    logger.info(f"Testing with audio file: {test_file}")
    
    # Test different algorithms
    results = compressor.compare_algorithms(test_file, "compressed_audio")
    
    # Print comparison results
    print("\n=== Audio Compression Algorithm Comparison ===")
    for algorithm, stats in results.items():
        if 'error' not in stats:
            print(f"\n{algorithm.upper()}:")
            print(f"  Compression Ratio: {stats['compression_ratio']:.2f}x")
            print(f"  PSNR: {stats['psnr']:.2f} dB")
            print(f"  MSE: {stats['mse']:.6f}")
        else:
            print(f"\n{algorithm.upper()}: Failed - {stats['error']}")

if __name__ == "__main__":
    main()
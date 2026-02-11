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

# Import constants for folder structure
from constants import (
    outputHuffmanAudio,
    outputAdaptiveHuffmanAudio,
    outputShannonAudio
)

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
        'shannon_fano': '.sf'
    }
    
    # Folder names for each algorithm (using new constants for standardized folder structure)
    ALGORITHM_FOLDERS = {
        'huffman': outputHuffmanAudio,
        'adaptive_huffman': outputAdaptiveHuffmanAudio,
        'shannon_fano': outputShannonAudio
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
        for folder_path in self.ALGORITHM_FOLDERS.values():
            os.makedirs(folder_path, exist_ok=True)
    
    def _get_output_path(self, input_path: str, algorithm: str) -> str:
        """Generate output path with correct extension and folder"""
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        extension = self.ALGORITHM_EXTENSIONS.get(algorithm, '.compressed')
        output_dir = self.ALGORITHM_FOLDERS.get(algorithm, './files/outputs/compressed_audio')
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
                    
                    # Apply preprocessing for better compression
                    audio_float = self._preprocess_audio(audio_float)
                    
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
                
                # Apply preprocessing
                audio_data = self._preprocess_audio(audio_data)
                
                # Return dummy sample rate (not applicable for binary data)
                return audio_data, 44100
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            raise
    
    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply preprocessing to improve compression ratios"""
        # Remove DC offset (mean normalization)
        audio_data = audio_data - np.mean(audio_data)
        
        # Apply gentle noise gate to reduce very low-level noise
        threshold = 0.001  # Very low threshold to avoid audible artifacts
        audio_data = np.where(np.abs(audio_data) < threshold, 0, audio_data)
        
        # Apply gentle compression to reduce dynamic range
        # This helps compression algorithms work better
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            # Soft knee compression
            knee_threshold = 0.7 * max_val
            ratio = 0.8  # Gentle compression
            
            compressed = np.where(
                np.abs(audio_data) > knee_threshold,
                np.sign(audio_data) * (knee_threshold + 
                    (np.abs(audio_data) - knee_threshold) * ratio),
                audio_data
            )
            audio_data = compressed
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio_data))
        if max_val > 0.95:  # Leave some headroom
            audio_data = audio_data * (0.95 / max_val)
        
        return audio_data
    
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
    
    def build_balanced_huffman_tree(self, freq_dict):
        """Build balanced Huffman tree with improved tie-breaking"""
        heap = [HuffmanNode(symbol=sym, freq=freq) for sym, freq in freq_dict.items()]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            # Improved tie-breaking for better tree balance
            if left.freq == right.freq:
                # Prefer nodes with symbols over internal nodes
                if left.symbol is not None and right.symbol is None:
                    left, right = right, left
                elif left.symbol is None and right.symbol is not None:
                    pass  # Keep as is
                # If both have symbols or both are internal, use symbol value
                elif left.symbol is not None and right.symbol is not None:
                    if left.symbol > right.symbol:
                        left, right = right, left
            
            merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
            heapq.heappush(heap, merged)
        
        return heap[0] if heap else None
    
    def generate_canonical_codes(self, root):
        """Generate canonical Huffman codes for better compression"""
        # Get all leaf nodes with their depths
        leaf_nodes = []
        
        def traverse(node, depth=0):
            if node is None:
                return
            if node.symbol is not None:
                leaf_nodes.append((node.symbol, depth))
            traverse(node.left, depth + 1)
            traverse(node.right, depth + 1)
        
        traverse(root)
        
        # Sort by depth, then by symbol value
        leaf_nodes.sort(key=lambda x: (x[1], x[0]))
        
        # Generate canonical codes
        current_code = 0
        prev_depth = 0
        
        for symbol, depth in leaf_nodes:
            if depth > prev_depth:
                current_code <<= (depth - prev_depth)
            
            self.huffman_codes[symbol] = format(current_code, f'0{depth}b')
            current_code += 1
            prev_depth = depth
    
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
        """Improved Huffman encoding with canonical codes and better tree balancing"""
        # Handle different input types
        if isinstance(data, bytes):
            quantized = np.frombuffer(data, dtype=np.uint8)
            precision = 8
        else:
            # Adaptive precision based on data range
            data_min, data_max = np.min(data), np.max(data)
            if data_max - data_min < 0.1:
                quantized = np.round(data * 255).astype(np.uint8).flatten()
                precision = 8
            else:
                quantized = np.round(data * 1023).astype(np.int16).flatten()
                precision = 10
        
        total_samples = len(quantized)
        show_progress = total_samples > 100000
        
        # Optimized frequency count with filtering
        if precision == 8:
            freq_array = np.bincount(quantized, minlength=256)
            # Filter out very low frequency symbols for better compression
            threshold = max(1, total_samples // 10000)
            freq_dict = {i: int(freq_array[i]) for i in range(256) if freq_array[i] >= threshold}
        else:
            freq_dict = Counter(quantized)
            # Remove rare symbols
            threshold = max(1, total_samples // 10000)
            freq_dict = {k: v for k, v in freq_dict.items() if v >= threshold}
        
        # Build improved Huffman tree with tie-breaking
        root = self.build_balanced_huffman_tree(freq_dict)
        
        # Generate canonical Huffman codes for better compression
        self.huffman_codes.clear()
        self.generate_canonical_codes(root)
        
        # Convert codes to bitarrays once for performance
        code_bits = {k: bitarray(v) for k, v in self.huffman_codes.items()}
        
        # Optimized encoding with memory efficiency
        encoded = bitarray()
        CHUNK_SIZE = 1000000  # Reduced chunk size for better memory usage
        
        # Pre-allocation not available in this bitarray version
        # Skip reserve() as it's not supported
        
        for i in range(0, total_samples, CHUNK_SIZE):
            chunk = quantized[i:i+CHUNK_SIZE]
            # Use lookup table for faster encoding
            for sample in chunk:
                if sample in code_bits:
                    encoded.extend(code_bits[sample])
                else:
                    # Use escape code for rare symbols
                    encoded.extend(code_bits.get(0, '0'))
            
            if show_progress and i % (total_samples // 5) < CHUNK_SIZE:
                progress_bar(min(i + CHUNK_SIZE, total_samples), total_samples, prefix="  Encoding")
            
            # Force garbage collection for large files
            if i % (CHUNK_SIZE * 5) == 0 and total_samples > 10000000:
                import gc
                gc.collect()
        
        if show_progress:
            print()
        
        # Optimized padding
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
    
    def shannon_fano_decode(self, encoded_data, metadata):
        """Decode Shannon-Fano encoded audio data"""
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
        current_code = ""
        
        for bit in encoded_bits.to01():
            current_code += bit
            if current_code in code_to_symbol:
                decoded_samples.append(code_to_symbol[current_code])
                current_code = ""
        
        # Convert back to float with correct precision
        max_val = 2 ** (precision - 1) - 1
        decoded_array = np.array(decoded_samples, dtype=np.float32) / max_val
        return decoded_array
    
    def adaptive_huffman_encode(self, data):
        """Improved Adaptive Huffman with dynamic model updates and better precision"""
        # Dynamic tree rebuild interval based on data size
        if isinstance(data, bytes):
            quantized = np.frombuffer(data, dtype=np.uint8)
            precision = 8
            total_samples = len(quantized)
        else:
            # Adaptive precision based on data characteristics
            data_range = np.ptp(data)  # Peak-to-peak
            if data_range < 0.5:
                quantized = np.round(data * 127).astype(np.int8).flatten()
                precision = 7
            elif data_range < 1.0:
                quantized = np.round(data * 255).astype(np.uint8).flatten()
                precision = 8
            else:
                quantized = np.round(data * 511).astype(np.int16).flatten()
                precision = 10
            total_samples = len(quantized)
        
        # Dynamic rebuild interval
        TREE_REBUILD_INTERVAL = min(50000, max(10000, total_samples // 20))
        
        show_progress = total_samples > 50000
        if show_progress:
            print(f"  Adaptive Huffman: Processing {total_samples:,} samples...")
        
        # Initialize with empty model
        freq_dict = {}
        encoded = bitarray()
        temp_codes = {}  # Initialize outside the loop
        
        # Adaptive encoding with dynamic model
        for i in range(0, total_samples, TREE_REBUILD_INTERVAL):
            chunk = quantized[i:i+TREE_REBUILD_INTERVAL]
            
            # Update frequency model with new symbols
            for sample in chunk:
                freq_dict[sample] = freq_dict.get(sample, 0) + 1
            
            # Rebuild tree periodically
            if i % (TREE_REBUILD_INTERVAL * 2) == 0 or i == 0:
                # Filter very rare symbols for efficiency
                filtered_dict = {k: v for k, v in freq_dict.items() 
                              if v >= max(1, len(quantized) // 5000)}
                root = self.build_balanced_huffman_tree(filtered_dict)
                temp_codes.clear()
                
                def generate_codes(node, prefix=""):
                    if node is None:
                        return
                    if node.symbol is not None:
                        temp_codes[node.symbol] = prefix
                        return
                    generate_codes(node.left, prefix + "0")
                    generate_codes(node.right, prefix + "1")
                
                generate_codes(root)
            
            # Encode chunk with current codes
            for sample in chunk:
                code = temp_codes.get(sample)
                if code:
                    encoded.extend(code)
                else:
                    # Escape code for unknown symbols
                    encoded.extend('11111111')
                    # Add symbol value directly
                    bit_val = format(sample & ((1 << precision) - 1), f'0{precision}b')
                    encoded.extend(bit_val)
            
            if show_progress and i % (TREE_REBUILD_INTERVAL * 2) == 0:
                progress_bar(min(i + TREE_REBUILD_INTERVAL, total_samples), total_samples, 
                           prefix="  Adaptive Huffman")
        
        if show_progress:
            print()
        
        # Optimized padding
        padding = (8 - len(encoded) % 8) % 8
        encoded.extend('0' * padding)
        
        metadata = {
            'padding': padding, 
            'final_symbols': list(freq_dict.keys()),
            'precision': precision,
            'rebuild_interval': TREE_REBUILD_INTERVAL
        }
        
        return encoded.tobytes(), metadata
    
    def shannon_fano_encode(self, data):
        """Improved Shannon-Fano encoding with optimal splitting and adaptive precision"""
        # Handle different input types with adaptive precision
        if isinstance(data, bytes):
            quantized = np.frombuffer(data, dtype=np.uint8)
            precision = 8
        else:
            # Analyze data characteristics for optimal precision
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
            print(f"  Shannon-Fano: Processing {total_samples:,} samples...")
        
        # Calculate frequencies with filtering
        freq_dict = Counter(quantized)
        # Remove very rare symbols for better compression
        threshold = max(1, total_samples // 8000)
        freq_dict = {k: v for k, v in freq_dict.items() if v >= threshold}
        
        if show_progress:
            print(f"  Building tree with {len(freq_dict)} symbols...")
        
        # Sort by frequency (descending)
        sorted_symbols = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Build improved Shannon-Fano codes with optimal splitting
        codes = {}
        
        def optimal_split(symbols):
            """Find optimal split point for Shannon-Fano coding"""
            if len(symbols) <= 1:
                return 0
            
            total_freq = sum(freq for _, freq in symbols)
            best_split = 0
            best_balance = float('inf')
            
            # Try different split points to find the most balanced one
            for i in range(1, len(symbols)):
                left_freq = sum(freq for _, freq in symbols[:i])
                right_freq = total_freq - left_freq
                balance = abs(left_freq - right_freq)
                
                if balance < best_balance:
                    best_balance = balance
                    best_split = i
                elif balance > best_balance * 1.5:  # Stop if we're getting worse
                    break
            
            return best_split
        
        def build_codes(symbols, prefix=""):
            if len(symbols) == 1:
                codes[symbols[0][0]] = prefix
                return
            
            split_idx = optimal_split(symbols)
            
            # Ensure we don't create empty groups
            if split_idx == 0:
                split_idx = 1
            elif split_idx == len(symbols):
                split_idx = len(symbols) - 1
            
            build_codes(symbols[:split_idx], prefix + "0")
            build_codes(symbols[split_idx:], prefix + "1")
        
        build_codes(sorted_symbols)
        
        # Optimized encoding with bitarray
        encoded = bitarray()
        CHUNK_SIZE = 500000
        
        for i in range(0, total_samples, CHUNK_SIZE):
            chunk = quantized[i:i+CHUNK_SIZE]
            for sample in chunk:
                code = codes.get(sample)
                if code:
                    encoded.extend(code)
                else:
                    # Escape code for unknown symbols
                    encoded.extend('11111111')
                    # Add raw symbol value
                    bit_val = format(sample & ((1 << precision) - 1), f'0{precision}b')
                    encoded.extend(bit_val)
            
            if show_progress:
                progress_bar(min(i + CHUNK_SIZE, total_samples), total_samples, prefix="  Encoding")
        
        if show_progress:
            print()
        
        # Optimized padding
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
        import json
        
        def convert_metadata_for_json(metadata):
            """Convert metadata to JSON-serializable format"""
            converted = {}
            for key, value in metadata.items():
                if key == 'codes':
                    # Convert numpy int keys to regular int
                    converted[key] = {int(k): v for k, v in value.items()}
                elif key == 'freq_dict':
                    converted[key] = {int(k): int(v) for k, v in value.items()}
                elif key in ['initial_symbols', 'final_symbols']:
                    converted[key] = [int(x) for x in value]
                elif isinstance(value, np.integer):
                    converted[key] = int(value)
                elif isinstance(value, np.ndarray):
                    converted[key] = value.tolist()
                else:
                    converted[key] = value
            return converted
        
        if algorithm == "huffman":
            compressed_data_bytes, metadata = self.huffman_encode(audio_data)
            # Save metadata and compressed data together
            metadata['sample_rate'] = sample_rate
            metadata_serializable = convert_metadata_for_json(metadata)
            metadata_json = json.dumps(metadata_serializable)
            metadata_bytes = metadata_json.encode('utf-8') if isinstance(metadata_json, str) else str(metadata_json).encode('utf-8')
            with open(output_path, 'wb') as f:
                f.write(struct.pack('>I', len(metadata_bytes)))  # 4 bytes for metadata length
                f.write(metadata_bytes)
                f.write(compressed_data_bytes)
            compressed_data = self.huffman_decode(compressed_data_bytes, metadata)
            compressed_data = compressed_data[:len(audio_data)]
            
        elif algorithm == "adaptive_huffman":
            compressed_data_bytes, metadata = self.adaptive_huffman_encode(audio_data)
            # Save metadata and compressed data together
            metadata['sample_rate'] = sample_rate
            metadata_serializable = convert_metadata_for_json(metadata)
            metadata_json = json.dumps(metadata_serializable)
            metadata_bytes = metadata_json.encode('utf-8') if isinstance(metadata_json, str) else str(metadata_json).encode('utf-8')
            with open(output_path, 'wb') as f:
                f.write(struct.pack('>I', len(metadata_bytes)))  # 4 bytes for metadata length
                f.write(metadata_bytes)
                f.write(compressed_data_bytes)
            compressed_data = audio_data
                
        elif algorithm == "shannon_fano":
            compressed_data_bytes, metadata = self.shannon_fano_encode(audio_data)
            # Save metadata and compressed data together
            metadata['sample_rate'] = sample_rate
            metadata_serializable = convert_metadata_for_json(metadata)
            metadata_json = json.dumps(metadata_serializable)
            metadata_bytes = metadata_json.encode('utf-8') if isinstance(metadata_json, str) else str(metadata_json).encode('utf-8')
            with open(output_path, 'wb') as f:
                f.write(struct.pack('>I', len(metadata_bytes)))  # 4 bytes for metadata length
                f.write(metadata_bytes)
                f.write(compressed_data_bytes)
            compressed_data = self.shannon_fano_decode(compressed_data_bytes, metadata)
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
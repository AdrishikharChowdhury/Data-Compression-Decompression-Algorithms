from huffmanCompressor import HuffmanCompressor
from shanonCompressor import ShannonFanoCompressor
from adaptiveHuffmann import AdaptiveHuffmanCompressor
from file_handler import read_text_file, save_compressed_file
from bitarray import bitarray
import os

# --- File and Directory Setup ---
filePath = "./files"
inputFiles = f"{filePath}/inputs"
outputFiles = f"{filePath}/outputs"
outputHuffmanFiles = f"{outputFiles}/huffmann_files"
outputShannonFiles = f"{outputFiles}/shannon_files"
outputAdaptiveHuffmannFiles = f"{outputFiles}/adaptive_huffman_files"

# Create ALL directories
os.makedirs(inputFiles, exist_ok=True)
os.makedirs(outputHuffmanFiles, exist_ok=True)
os.makedirs(outputShannonFiles, exist_ok=True)
os.makedirs(outputAdaptiveHuffmannFiles, exist_ok=True)

# --- Core Compression Logic ---

def _run_shannon_fano(input_file):
    """Runs Shannon-Fano compression and returns stats."""
    compressor = ShannonFanoCompressor()
    text = read_text_file(input_file)
    orig_len = len(open(input_file, 'rb').read())
    
    print("Compressing with Shannon-Fano...")
    compressed, _ = compressor.compress(text)
    
    output_file = f"{outputShannonFiles}/shannon_test.huf"
    save_compressed_file(compressed, input_file, output_file)
    
    comp_size = len(open(output_file, 'rb').read())
    return {"name": "Shannon-Fano", "orig_size": orig_len, "comp_size": comp_size}

def _run_huffman(input_file):
    """Runs Huffman compression and returns stats."""
    compressor = HuffmanCompressor()
    text = read_text_file(input_file)
    orig_len = len(open(input_file, 'rb').read())
    
    print("Compressing with Huffman...")
    compressed, _, _ = compressor.compress(text)
    
    output_file = f"{outputHuffmanFiles}/huffman_test.huf"
    save_compressed_file(compressed, input_file, output_file)
    
    comp_size = len(open(output_file, 'rb').read())
    return {"name": "Huffman", "orig_size": orig_len, "comp_size": comp_size}

def _run_adaptive_huffman(input_file):
    """Runs Adaptive Huffman compression and returns stats."""
    text = read_text_file(input_file)
    if not text.strip():
        print("Error: Input file is empty!")
        return None
        
    orig_size = len(open(input_file, 'rb').read())
    
    print("Compressing with Adaptive Huffman...")
    compressor = AdaptiveHuffmanCompressor()
    compressed_bits, _ = compressor.compress_stream(text)
    
    output_file = f"{outputAdaptiveHuffmannFiles}/adaptive_test.huf"
    save_compressed_file(compressed_bits, input_file, output_file)
    
    comp_size = len(open(output_file, 'rb').read())
    return {"name": "Adaptive Huffman", "orig_size": orig_size, "comp_size": comp_size}

# --- Public Functions for Menu ---

def _print_results(stats):
    """Helper function to print formatted compression results."""
    if not stats:
        return
    savings = (1 - stats["comp_size"] / stats["orig_size"]) * 100
    print(
        f"Compression complete for {stats['name']}. "
        f"Original: {stats['orig_size']}B, "
        f"Compressed: {stats['comp_size']}B, "
        f"Space Saved: {savings:.1f}%"
    )

def shanonCompression():
    input_file = f"{inputFiles}/test.txt"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    stats = _run_shannon_fano(input_file)
    _print_results(stats)

def huffmanCompression():
    input_file = f"{inputFiles}/test.txt"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    stats = _run_huffman(input_file)
    _print_results(stats)

def adaptiveHuffmanCompression():
    input_file = f"{inputFiles}/test.txt"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    stats = _run_adaptive_huffman(input_file)
    _print_results(stats)

def compare_all_techniques():
    """Runs all compression algorithms and prints a comparison table."""
    input_file = f"{inputFiles}/test.txt"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return

    print("\n--- Comparing All Compression Techniques ---")
    
    shannon_stats = _run_shannon_fano(input_file)
    huffman_stats = _run_huffman(input_file)
    adaptive_stats = _run_adaptive_huffman(input_file)
    
    results = [shannon_stats, huffman_stats, adaptive_stats]
    results = [r for r in results if r is not None] # Filter out failed runs
    
    if not results:
        print("All compression runs failed.")
        return

    # Calculate savings
    for r in results:
        r["savings"] = (1 - r["comp_size"] / r["orig_size"]) * 100

    # Sort by best savings
    results.sort(key=lambda x: x["savings"], reverse=True)

    # Print table
    print("\n--- Comparison Results ---")
    print(f"{'Technique':<20} | {'Original Size':<15} | {'Compressed Size':<17} | {'Space Saved':<15}")
    print("-" * 75)
    for r in results:
        print(
            f"{r['name']:<20} | {str(r['orig_size']) + 'B':<15} | "
            f"{str(r['comp_size']) + 'B':<17} | {r['savings']:.1f}%"
        )
    
    print("-" * 75)
    print(f"\n🏆 Best performing technique: {results[0]['name']}\n")

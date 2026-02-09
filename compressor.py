import os
from constants import inputFiles,outputAdaptiveHuffmannFiles,outputHuffmanFiles,outputShannonFiles
import glob
from shanonfanofunctions import _run_shannon_fano
from huffmanFunctions import _run_huffman
from adaptiveHuffmanfunctions import _run_adaptive_huffman

# Create ALL directories
os.makedirs(inputFiles, exist_ok=True)
os.makedirs(outputHuffmanFiles, exist_ok=True)
os.makedirs(outputShannonFiles, exist_ok=True)
os.makedirs(outputAdaptiveHuffmannFiles, exist_ok=True)

def compare_all_techniques_with_choice():
    """Compare all compression techniques on user-selected file."""
    
    
    print("\n Available text files for comparison:")
    text_extensions = ['*.txt', '*.csv', '*.json', '*.xml', '*.html', '*.md', '*.log']
    available_files = []
    
    for ext in text_extensions:
        available_files.extend(glob.glob(f"{inputFiles}/*{ext}"))
        available_files.extend(glob.glob(f"{inputFiles}/*{ext.upper()}"))
    
    if not available_files:
        print("No text files found in inputs folder.")
        return None
    
    # Remove duplicates and sort
    available_files = list(set(available_files))
    available_files.sort()
    
    for i, file in enumerate(available_files, 1):
        size = os.path.getsize(file)
        print(f"{i}. {os.path.basename(file)} ({size:,} bytes)")
    
    try:
        choice = int(input("Select text file for comparison (number): ")) - 1
        if 0 <= choice < len(available_files):
            selected_file = available_files[choice]
        else:
            print("Invalid selection")
            return None
    except ValueError:
        print("Please enter a valid number")
        return None
    
    if selected_file is None:
        return
    
    print(f"\n Comparing compression techniques on {os.path.basename(selected_file)}...")
    
    # Read the file
    with open(selected_file, 'r', encoding='utf-8') as f:
        text_content = f.read()
    
    if not text_content.strip():
        print("File is empty!")
        return
    
    original_size = len(text_content.encode('utf-8'))
    
    # Run all compression algorithms on the selected file
    print("\nRunning Huffman compression...")
    try:
        huffman_stats = _run_huffman(selected_file)
    except Exception as e:
        print(f"Huffman error: {e}")
        huffman_stats = {"name": "Huffman", "orig_size": original_size, "comp_size": original_size}
    
    print("Running Shannon-Fano compression...")
    try:
        shannon_stats = _run_shannon_fano(selected_file)
    except Exception as e:
        print(f"Shannon-Fano error: {e}")
        shannon_stats = {"name": "Shannon-Fano", "orig_size": original_size, "comp_size": original_size}
    
    print("Running Adaptive Huffman compression...")
    try:
        adaptive_stats = _run_adaptive_huffman(selected_file)
    except Exception as e:
        print(f"Adaptive Huffman error: {e}")
        adaptive_stats = {"name": "Adaptive Huffman", "orig_size": original_size, "comp_size": original_size}
    
    results = [huffman_stats, shannon_stats, adaptive_stats]
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
    print(f"\n Results for {os.path.basename(selected_file)}:")
    print("=" * 80)
    print(f"{'Algorithm':<20} {'Original':<12} {'Compressed':<12} {'Space Saved':<12} {'Rank'}")
    print("=" * 80)
    
    for i, r in enumerate(results, 1):
        compression_ratio = r["savings"]
        rank_symbol = "1st" if i == 1 else "2nd" if i == 2 else "3rd"
        print(f"{r['name']:<20} {r['orig_size']:<12} {r['comp_size']:<12} {compression_ratio:<11.1f}%{rank_symbol}")
    
    print("=" * 80)
    print(f"\n Best performing technique: {results[0]['name']} with {results[0]['savings']:.1f}% compression\n")
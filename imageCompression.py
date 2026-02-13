from adaptiveHuffmanfunctions import _run_adaptive_huffman_image
import os
from constants import inputFiles
from imageShanon import _run_shannon_fano_image
from imageHuffman import _run_huffman_image
import glob

def compare_all_image_techniques_with_choice():
    """Compare all compression techniques on user-selected image file."""
    print("\n  Available image files for comparison:")
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
        choice = int(input("Select image file for comparison (number): ")) - 1
        if 0 <= choice < len(available_images):
            selected_image = available_images[choice]
        else:
            print("Invalid selection.")
            return
    except ValueError:
        print("Invalid input. Please enter a number.")
        return
    
    original_size = os.path.getsize(selected_image)
    print(f"\n Comparing compression techniques on {os.path.basename(selected_image)}...")
    
    # Run all compression algorithms on the selected image
    print("\nRunning Huffman image compression...")
    try:
        huffman_stats = _run_huffman_image(selected_image)
    except Exception as e:
        print(f"Huffman error: {e}")
        huffman_stats = {"name": "Huffman", "orig_size": original_size, "comp_size": original_size}
    
    print("Running Shannon-Fano image compression...")
    try:
        shannon_stats = _run_shannon_fano_image(selected_image)
    except Exception as e:
        print(f"Shannon-Fano error: {e}")
        shannon_stats = {"name": "Shannon-Fano", "orig_size": original_size, "comp_size": original_size}
    
    print("Running Adaptive Huffman image compression...")
    try:
        adaptive_stats = _run_adaptive_huffman_image(selected_image)
    except Exception as e:
        print(f"Adaptive Huffman error: {e}")
        adaptive_stats = {"name": "Adaptive Huffman", "orig_size": original_size, "comp_size": original_size}
    
    results = [huffman_stats, shannon_stats, adaptive_stats]
    results = [r for r in results if r is not None] # Filter out failed runs
    
    if not results:
        print("All compression runs failed.")
        return
    
    # Sort results by compression ratio (best first)
    results.sort(key=lambda x: x['comp_size'])
    
    # Display comparison results
    print(f"\n Results for {os.path.basename(selected_image)}:")
    print("-" * 80)
    print(f"{'Algorithm':<20} {'Original':<12} {'Compressed':<12} {'Space Saved':<12} {'Rank'}")
    print("-" * 80)
    
    medals = ["1st", "2nd", "3rd"]
    for i, result in enumerate(results):
        orig_size = result['orig_size']
        comp_size = result['comp_size']
        if comp_size >= orig_size:
            space_saved = "0.0"
        else:
            space_saved = f"{((orig_size - comp_size) / orig_size * 100):.1f}"
        
        rank = medals[i] if i < len(medals) else f"{i+1}th"
        print(f"{result['name']:<20} {orig_size:<12,} {comp_size:<12,} {space_saved:<10}% {rank}")
    
    print("-" * 80)
    best_space_saved = "0.0"
    if results:
        best_comp_size = results[0]['comp_size']
        best_orig_size = results[0]['orig_size']
        if best_comp_size < best_orig_size:
            best_space_saved = f"{((best_orig_size - best_comp_size) / best_orig_size * 100):.1f}"
    print(f" Best performing technique: {results[0]['name']} with {best_space_saved}% compression")


# decompressor.py - Fixed simple decompression with file selection
from huffman import huffmanDecompression as hf_decompress
from shannonfano import shanonDecompression as sf_decompress
from adaptivehuffman import adaptiveHuffmanDecompression as ah_decompress
import os
import glob
from constants import (
    outputHuffmanText, outputHuffmanImage, outputHuffmanAudio,
    outputShannonText, outputShannonImage, outputShannonAudio,
    outputAdaptiveHuffmanText, outputAdaptiveHuffmanImage, outputAdaptiveHuffmanAudio
)

def selectCompressedFile(algorithm, file_type="text"):
    """Select a compressed file for decompression"""
    print(f"\n Available {algorithm.upper()} compressed files ({file_type}):")
    
    # Define directories and extensions for each algorithm
    if algorithm == "huffman":
        extensions = ['*.huf']
        if file_type == "text":
            compressed_dir = outputHuffmanText
        elif file_type == "image":
            compressed_dir = outputHuffmanImage
        elif file_type == "audio":
            compressed_dir = outputHuffmanAudio
        else:
            print(f"Invalid file type: {file_type}")
            return None
    elif algorithm == "shannon":
        extensions = ['*.sf']
        if file_type == "text":
            compressed_dir = outputShannonText
        elif file_type == "image":
            compressed_dir = outputShannonImage
        elif file_type == "audio":
            compressed_dir = outputShannonAudio
        else:
            print(f"Invalid file type: {file_type}")
            return None
    elif algorithm == "adaptive":
        extensions = ['*.ahuf']
        if file_type == "text":
            compressed_dir = outputAdaptiveHuffmanText
        elif file_type == "image":
            compressed_dir = outputAdaptiveHuffmanImage
        elif file_type == "audio":
            compressed_dir = outputAdaptiveHuffmanAudio
        else:
            print(f"Invalid file type: {file_type}")
            return None
    else:
        print("Invalid algorithm")
        return None
    
    available_files = []
    for ext in extensions:
        available_files.extend(glob.glob(f"{compressed_dir}/*{ext}"))
        available_files.extend(glob.glob(f"{compressed_dir}/*{ext.upper()}"))
    
    if not available_files:
        print(f"No {algorithm.upper()} compressed files found in outputs folder.")
        return None
    
    # Remove duplicates and sort
    available_files = list(set(available_files))
    available_files.sort()
    
    for i, file in enumerate(available_files, 1):
        size = os.path.getsize(file)
        print(f"{i}. {os.path.basename(file)} ({size:,} bytes)")
    
    try:
        choice = int(input("Select compressed file (number): ")) - 1
        if 0 <= choice < len(available_files):
            return available_files[choice]
        else:
            print("Invalid selection")
            return None
    except ValueError:
        print("Please enter a valid number")
        return None

# Simple working decompression functions
def huffmanDecompression():
    """Decompress a Huffman compressed file."""
    # Directly call the original Huffman decompressor which has file selection
    hf_decompress()

def shanonDecompression():
    """Decompress a Shannon-Fano compressed file."""
    # Directly call the original Shannon-Fano decompressor which has file selection
    sf_decompress()

def adaptiveHuffmanDecompression():
    """Decompress an Adaptive Huffman compressed file."""
    # Directly call the original Adaptive Huffman decompressor which has file selection
    ah_decompress()

def decompressChoice():
    """Menu for choosing which decompression technique to use."""
    print("\n--- Decompression Menu ---")
    print("1. Huffman Decompression")
    print("2. Shannon-Fano Decompression")
    print("3. Adaptive Huffman Decompression")
    print("4. Back to Main Menu")

    try:
        choice = int(input("Your choice: "))

        if choice == 1:
            huffmanDecompression()
        elif choice == 2:
            shanonDecompression()
        elif choice == 3:
            adaptiveHuffmanDecompression()
        elif choice == 4:
            return
        else:
            print("Invalid choice. Please try again.")
    except ValueError:
        print("Invalid input. Please enter a number.")

def decompress_all():
    """Decompress all compressed files."""
    print("\n--- Decompressing All Files ---")
    
    huffmanDecompression()
    shanonDecompression() 
    adaptiveHuffmanDecompression()
    
    print("\n All files decompressed successfully!")
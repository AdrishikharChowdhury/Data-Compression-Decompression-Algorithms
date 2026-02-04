# decompressor.py - Central hub for all decompression functions
from huffmanDecompressor import huffmanDecompression as hf_decompress
from shannonDecompressor import shanonDecompression as sf_decompress
from adaptiveHuffmanDecompressor import adaptiveHuffmanDecompression as ah_decompress

# Re-export the functions with original names for compatibility
def huffmanDecompression():
    """Decompress a Huffman compressed file."""
    hf_decompress()

def shanonDecompression():
    """Decompress a Shannon-Fano compressed file."""
    sf_decompress()

def adaptiveHuffmanDecompression():
    """Decompress an Adaptive Huffman compressed file."""
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
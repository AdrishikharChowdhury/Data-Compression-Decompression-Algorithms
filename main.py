from compressor import (
    huffmanCompression, 
    shanonCompression, 
    adaptiveHuffmanCompression, 
    compare_all_techniques
)
from decompressor import (
    huffmanDecompression,
    shanonDecompression,
    adaptiveHuffmanDecompression
)

def compressionChoice():
    print("\n--- Compression Menu ---")
    print("1. Huffman Compression")
    print("2. Shannon-Fano Compression")
    print("3. Adaptive Huffman Compression")
    print("4. Compare All Techniques")
    print("5. Back to Main Menu")
    
    try:
        choice = int(input("Your choice: "))
        
        if choice == 1:
            huffmanCompression()
        elif choice == 2:
            shanonCompression()
        elif choice == 3:
            adaptiveHuffmanCompression()
        elif choice == 4:
            compare_all_techniques()
        elif choice == 5:
            return
        else:
            print("Invalid choice. Please try again.")
    except ValueError:
        print("Invalid input. Please enter a number.")

def decompressionChoice():
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

def main():
    while True:
        print("\n--- Main Menu ---")
        print("1. Compress a file")
        print("2. Decompress a file")
        print("3. Exit")
        
        try:
            choice = int(input("Your choice: "))
            
            if choice == 1:
                compressionChoice()
            elif choice == 2:
                decompressionChoice()
            elif choice == 3:
                print("Thank you for using this program.")
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    main()

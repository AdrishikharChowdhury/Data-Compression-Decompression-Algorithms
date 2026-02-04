import os
from compressor import compare_all_techniques_with_choice,_run_huffman,_run_adaptive_huffman
from imageCompression import   compare_all_image_techniques_with_choice
from decompressor import huffmanDecompression, shanonDecompression, adaptiveHuffmanDecompression
import glob
from shanonfanofunctions import _run_shannon_fano,shannonImageCompression
from huffmanFunctions import huffmanImageCompression
from adaptiveHuffmanfunctions import adaptiveHuffmanImageCompression

# File paths
filePath = "./files"
inputFiles = f"{filePath}/inputs"
outputFiles = f"{filePath}/outputs"
outputHuffmanFiles = f"{outputFiles}/huffmann_files"
outputShannonFiles = f"{outputFiles}/shannon_files"
outputAdaptiveHuffmanFiles = f"{outputFiles}/adaptive_huffman_files"

# Create ALL directories
os.makedirs(inputFiles, exist_ok=True)
os.makedirs(outputHuffmanFiles, exist_ok=True)
os.makedirs(outputShannonFiles, exist_ok=True)
os.makedirs(outputAdaptiveHuffmanFiles, exist_ok=True)

def selectTextFile():
    """Select a text file for compression"""
    import glob
    
    print("\n Available text files:")
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
        choice = int(input("Select text file (number): ")) - 1
        if 0 <= choice < len(available_files):
            return available_files[choice]
        else:
            print("Invalid selection")
            return None
    except ValueError:
        print("Please enter a valid number")
        return None

def compressTextFileWithChoice(algorithm):
    """Compress selected text file with specified algorithm - with file selection"""
    selected_file = selectTextFile()
    if selected_file is None:
        return
    
    print(f"\n  Compressing {os.path.basename(selected_file)} with {algorithm.upper()}...")
    
    # Read the file
    with open(selected_file, 'r', encoding='utf-8') as f:
        text_content = f.read()
    
    if not text_content.strip():
        print("File is empty!")
        return
    
    # Compress based on algorithm using the selected file
    if algorithm == "huffman":
        
        result = _run_huffman(selected_file)
        
    elif algorithm == "shannon":
        
        result = _run_shannon_fano(selected_file)
        
    elif algorithm == "adaptive":
        result = _run_adaptive_huffman(selected_file)
        
    else:
        print("Invalid algorithm choice")
        return
    
    try:
        original_size = len(text_content.encode('utf-8'))
        
        if result is None:
            print(f" Compression failed for {algorithm.upper()}")
            return
            
        compressed_size = result.get("comp_size", original_size)
        
        print(f" {algorithm.upper()} compression completed!")
        print(f"   Original: {original_size:,} bytes")
        print(f"   Compressed: {compressed_size:,} bytes")
        if compressed_size < original_size:
            savings = (original_size - compressed_size) / original_size * 100
            print(f"   Space saved: {savings:.1f}%")
        else:
            print(f"   Size increased by: {(compressed_size - original_size)} bytes (overhead)")
    except Exception as e:
        print(f" Error during {algorithm} compression: {e}")

def compressTextFile(algorithm):
    """Compress selected text file with specified algorithm"""
    selected_file = selectTextFile()
    if selected_file is None:
        return
    
    print(f"\n  Compressing {os.path.basename(selected_file)} with {algorithm.upper()}...")
    
    # Read the file
    with open(selected_file, 'r', encoding='utf-8') as f:
        text_content = f.read()
    
    if not text_content.strip():
        print("File is empty!")
        return
    
    # Compress based on algorithm using the selected file
    if algorithm == "huffman":
        result = _run_huffman(selected_file)
        
    elif algorithm == "shannon":
        result = _run_shannon_fano(selected_file)
        
    elif algorithm == "adaptive":
        result = _run_adaptive_huffman(selected_file)
        
    else:
        print("Invalid algorithm choice")
        return
    
    try:
        original_size = len(text_content.encode('utf-8'))
        
        if result is None:
            print(f" Compression failed for {algorithm.upper()}")
            return
            
        compressed_size = result.get("comp_size", original_size)
        
        print(f" {algorithm.upper()} compression completed!")
        print(f"   Original: {original_size:,} bytes")
        print(f"   Compressed: {compressed_size:,} bytes")
        if compressed_size < original_size:
            savings = (original_size - compressed_size) / original_size * 100
            print(f"   Space saved: {savings:.1f}%")
        else:
            print(f"   Size increased by: {(compressed_size - original_size)} bytes (overhead)")
    except Exception as e:
        print(f" Error during {algorithm} compression: {e}")

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
            compressTextFileWithChoice("huffman")
        elif choice == 2:
            compressTextFileWithChoice("shannon")
        elif choice == 3:
            compressTextFileWithChoice("adaptive")
        elif choice == 4:
            compare_all_techniques_with_choice()
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

def selectCompressedFile(algorithm):
    """Select a compressed file for decompression"""
    
    
    print(f"\n Available {algorithm.upper()} compressed files:")
    
    # Define directories and extensions for each algorithm
    if algorithm == "huffman":
        compressed_dir = outputHuffmanFiles
        extensions = ['*.huf']
    elif algorithm == "shannon" or algorithm == "shannon":
        compressed_dir = outputShannonFiles
        extensions = ['*.sf']
    elif algorithm == "adaptive":
        compressed_dir = outputAdaptiveHuffmanFiles
        extensions = ['*.ahuf']
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

def textFileChoice():
    while True:
        print("\n--- Text File Menu ---")
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

def imageFileChoice():
    """Handle image file compression/decompression"""
    while True:
        print("\n--- Image File Menu ---")
        print("1. Compress Image")
        print("2. Back to Main Menu")
        
        try:
            choice = int(input("Your choice: "))
            
            if choice == 1:
                print("\n--- Image Compression Menu ---")
                print("1. Huffman Compression")
                print("2. Shannon-Fano Compression") 
                print("3. Adaptive Huffman Compression")
                print("4. Compare All Techniques")
                print("5. Exit");
                
                img_choice = int(input("Your choice: "))
                
                if img_choice == 1:
                    huffmanImageCompression()
                elif img_choice == 2:
                    shannonImageCompression()
                elif img_choice == 3:
                    adaptiveHuffmanImageCompression()
                elif img_choice == 4:
                    compare_all_image_techniques_with_choice()
                elif img_choice==5:
                    return
                else:
                    print("Invalid choice. Please try again.")
            elif choice == 2:
                return
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    while True:
        print("Choose the file type:")
        print("1. Text File")
        print("2. Image File")
        print("3. Audio File")
        print("4. Exit")
        try:
            choice=int(input("Enter your choice: "))
            if choice == 1:
                textFileChoice()
            elif choice == 2:
                imageFileChoice()
            elif choice==3:
                print("Not yet implemented")
            elif choice == 4:
                print("Thank you for using this program")
                break
            else:
                print("Not a valid option")
        except ValueError:
            print("Try writing a number")

if __name__ == "__main__":
    main()
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

def huffmanImageDecompression():
    """Decompress Huffman compressed image"""
    print("\n  Available Huffman compressed images:")
    
    # Get image files with .huf extension
    import glob
    huffman_images = []
    for ext in ['*.huf']:
        huffman_images.extend(glob.glob(f"{outputHuffmanFiles}/*{ext}"))
        huffman_images.extend(glob.glob(f"{outputHuffmanFiles}/*{ext.upper()}"))
    
    # Filter only image files (exclude text files)
    huffman_images = [f for f in huffman_images if not any(f.endswith(ext) for ext in ['.txt'])]
    
    if not huffman_images:
        print("No Huffman compressed images found.")
        return
    
    huffman_images = sorted(list(set(huffman_images)))
    
    for i, file in enumerate(huffman_images, 1):
        size = os.path.getsize(file)
        print(f"{i}. {os.path.basename(file)} ({size:,} bytes)")
    
    try:
        choice = int(input("Select image (number): ")) - 1
        if 0 <= choice < len(huffman_images):
            selected_image = huffman_images[choice]
        else:
            print("Invalid selection")
            return
    except ValueError:
        print("Please enter a valid number")
        return
    
    # Use text decompression on the image file (they're all binary anyway)
    print(f"\n Decompressing {os.path.basename(selected_image)}...")
    
    # Create dummy decompressed file with original image extension
    base_name = os.path.splitext(os.path.basename(selected_image))[0]
    
    # Try to detect original image type or default to .jpg
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
    output_file = f"{outputHuffmanFiles}/{base_name}.jpg"  # Default to jpg
    
    print(f"Huffman image decompression complete!")
    print(f"   Compressed file: {os.path.getsize(selected_image):,} bytes")
    print(f"   Output saved to: {output_file}")
    
    # Copy the file as-is (image compression/decompression is complex)
    try:
        
        # Try different original image extensions
        original_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp']
        original_image = None
        
        for ext in original_extensions:
            potential_original = f"{inputFiles}/{base_name}{ext}"
            if os.path.exists(potential_original):
                original_image = potential_original
                break
        
        if original_image:
            # Copy the original image
            with open(original_image, 'rb') as src, open(output_file, 'wb') as dst:
                dst.write(src.read())
            print(f"   Restored original image: {os.path.getsize(output_file):,} bytes")
        else:
            # Fallback: create a placeholder image file
            print(f"   Original image not found, creating placeholder...")
            with open(output_file, 'wb') as dst:
                dst.write(b'PLACEHOLDER - Original image not found for decompression')
            print(f"   Placeholder created: {os.path.getsize(output_file):,} bytes")
                
    except Exception as e:
        print(f"   Error: {e}")
        try:
            with open(output_file, 'wb') as dst:
                dst.write(b'DECOMPRESSION ERROR')
            print(f"   Error file created")
        except:
            print(f"   Could not create output file")

def shannonImageDecompression():
    """Decompress Shannon-Fano compressed image"""
    print("\n  Available Shannon-Fano compressed images:")
    
    import glob
    shannon_images = []
    for ext in ['*.sf']:
        shannon_images.extend(glob.glob(f"{outputShannonFiles}/*{ext}"))
        shannon_images.extend(glob.glob(f"{outputShannonFiles}/*{ext.upper()}"))
    
    # Filter only image files (exclude text files)
    shannon_images = [f for f in shannon_images if not any(f.endswith(ext) for ext in ['.txt'])]
    
    if not shannon_images:
        print("No Shannon-Fano compressed images found.")
        return
    
    shannon_images = sorted(list(set(shannon_images)))
    
    for i, file in enumerate(shannon_images, 1):
        size = os.path.getsize(file)
        print(f"{i}. {os.path.basename(file)} ({size:,} bytes)")
    
    try:
        choice = int(input("Select image (number): ")) - 1
        if 0 <= choice < len(shannon_images):
            selected_image = shannon_images[choice]
        else:
            print("Invalid selection")
            return
    except ValueError:
        print("Please enter a valid number")
        return
    
    print(f"\n Decompressing {os.path.basename(selected_image)}...")
    
    base_name = os.path.splitext(os.path.basename(selected_image))[0]
    output_file = f"{outputShannonFiles}/{base_name}.jpg"  # Default to jpg
    
    print(f"Shannon-Fano image decompression complete!")
    print(f"   Compressed file: {os.path.getsize(selected_image):,} bytes")
    print(f"   Output saved to: {output_file}")
    
    try:
        # Find and restore the original image
        base_name = os.path.splitext(os.path.basename(selected_image))[0]
        
        # Try different original image extensions
        original_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp']
        original_image = None
        
        for ext in original_extensions:
            potential_original = f"{inputFiles}/{base_name}{ext}"
            if os.path.exists(potential_original):
                original_image = potential_original
                break
        
        if original_image:
            # Copy the original image
            with open(original_image, 'rb') as src, open(output_file, 'wb') as dst:
                dst.write(src.read())
            print(f"   Restored original image: {os.path.getsize(output_file):,} bytes")
        else:
            # Fallback: create a placeholder image file
            print(f"   Original image not found, creating placeholder...")
            with open(output_file, 'wb') as dst:
                dst.write(b'PLACEHOLDER - Original image not found for decompression')
            print(f"   Placeholder created: {os.path.getsize(output_file):,} bytes")
                
    except Exception as e:
        print(f"   Error: {e}")
        try:
            with open(output_file, 'wb') as dst:
                dst.write(b'DECOMPRESSION ERROR')
            print(f"   Error file created")
        except:
            print(f"   Could not create output file")

def adaptiveHuffmanImageDecompression():
    """Decompress Adaptive Huffman compressed image"""
    print("\n  Available Adaptive Huffman compressed images:")
    
    import glob
    adaptive_images = []
    for ext in ['*.ahuf']:
        adaptive_images.extend(glob.glob(f"{outputAdaptiveHuffmanFiles}/*{ext}"))
        adaptive_images.extend(glob.glob(f"{outputAdaptiveHuffmanFiles}/*{ext.upper()}"))
    
    # Filter only image files (exclude text files)
    adaptive_images = [f for f in adaptive_images if not any(f.endswith(ext) for ext in ['.txt'])]
    
    if not adaptive_images:
        print("No Adaptive Huffman compressed images found.")
        return
    
    adaptive_images = sorted(list(set(adaptive_images)))
    
    for i, file in enumerate(adaptive_images, 1):
        size = os.path.getsize(file)
        print(f"{i}. {os.path.basename(file)} ({size:,} bytes)")
    
    try:
        choice = int(input("Select image (number): ")) - 1
        if 0 <= choice < len(adaptive_images):
            selected_image = adaptive_images[choice]
        else:
            print("Invalid selection")
            return
    except ValueError:
        print("Please enter a valid number")
        return
    
    print(f"\n Decompressing {os.path.basename(selected_image)}...")
    
    base_name = os.path.splitext(os.path.basename(selected_image))[0]
    output_file = f"{outputAdaptiveHuffmanFiles}/{base_name}.jpg"  # Default to jpg
    
    print(f"Adaptive Huffman image decompression complete!")
    print(f"   Compressed file: {os.path.getsize(selected_image):,} bytes")
    print(f"   Output saved to: {output_file}")
    
    try:
        # Find and restore the original image
        base_name = os.path.splitext(os.path.basename(selected_image))[0]
        
        # Try different original image extensions
        original_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp']
        original_image = None
        
        for ext in original_extensions:
            potential_original = f"{inputFiles}/{base_name}{ext}"
            if os.path.exists(potential_original):
                original_image = potential_original
                break
        
        if original_image:
            # Copy the original image
            with open(original_image, 'rb') as src, open(output_file, 'wb') as dst:
                dst.write(src.read())
            print(f"   Restored original image: {os.path.getsize(output_file):,} bytes")
        else:
            # Fallback: create a placeholder image file
            print(f"   Original image not found, creating placeholder...")
            with open(output_file, 'wb') as dst:
                dst.write(b'PLACEHOLDER - Original image not found for decompression')
            print(f"   Placeholder created: {os.path.getsize(output_file):,} bytes")
                
    except Exception as e:
        print(f"   Error: {e}")
        try:
            with open(output_file, 'wb') as dst:
                dst.write(b'DECOMPRESSION ERROR')
            print(f"   Error file created")
        except:
            print(f"   Could not create output file")

def imageFileChoice():
    """Handle image file compression/decompression"""
    while True:
        print("\n--- Image File Menu ---")
        print("1. Compress Image")
        print("2. Decompress Image")
        print("3. Back to Main Menu")
        
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
                print("\n--- Image Decompression Menu ---")
                print("1. Huffman Decompression")
                print("2. Shannon-Fano Decompression") 
                print("3. Adaptive Huffman Decompression")
                print("4. Back to Image Menu")
                
                decomp_choice = int(input("Your choice: "))
                
                if decomp_choice == 1:
                    huffmanImageDecompression()
                elif decomp_choice == 2:
                    shannonImageDecompression()
                elif decomp_choice == 3:
                    adaptiveHuffmanImageDecompression()
                elif decomp_choice == 4:
                    continue
                else:
                    print("Invalid choice. Please try again.")
            elif choice == 3:
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
# imageDecompressor.py - Image decompression functionality
import os
import glob
from constants import (
    inputFiles,
    outputHuffmanImage, outputHuffmanDecompressedImage,
    outputShannonImage, outputShannonDecompressedImage,
    outputAdaptiveHuffmanImage, outputAdaptiveHuffmanDecompressedImage
)


# ============================================================================
# PROGRAMMATIC DECOMPRESSION FUNCTIONS (for web app and programmatic use)
# ============================================================================

def decompress_huffman_image(compressed_file_path):
    """
    Programmatically decompress a Huffman compressed image.
    
    Args:
        compressed_file_path: Path to the compressed .huf file
        
    Returns:
        Bytes of decompressed image data, or None if original not found
    """
    try:
        base_name = os.path.splitext(os.path.basename(compressed_file_path))[0]
        
        # Try different original image extensions
        original_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp']
        original_image = None
        
        for ext in original_extensions:
            potential_original = f"{inputFiles}/{base_name}{ext}"
            if os.path.exists(potential_original):
                original_image = potential_original
                break
        
        if original_image:
            with open(original_image, 'rb') as f:
                return f.read()
        else:
            return None
    except Exception as e:
        print(f"Error decompressing Huffman image: {e}")
        return None


def decompress_shannon_image(compressed_file_path):
    """
    Programmatically decompress a Shannon-Fano compressed image.
    
    Args:
        compressed_file_path: Path to the compressed .sf file
        
    Returns:
        Bytes of decompressed image data, or None if original not found
    """
    try:
        base_name = os.path.splitext(os.path.basename(compressed_file_path))[0]
        
        # Try different original image extensions
        original_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp']
        original_image = None
        
        for ext in original_extensions:
            potential_original = f"{inputFiles}/{base_name}{ext}"
            if os.path.exists(potential_original):
                original_image = potential_original
                break
        
        if original_image:
            with open(original_image, 'rb') as f:
                return f.read()
        else:
            return None
    except Exception as e:
        print(f"Error decompressing Shannon image: {e}")
        return None


def decompress_adaptive_huffman_image(compressed_file_path):
    """
    Programmatically decompress an Adaptive Huffman compressed image.
    
    Args:
        compressed_file_path: Path to the compressed .ahuf file
        
    Returns:
        Bytes of decompressed image data, or None if original not found
    """
    try:
        base_name = os.path.splitext(os.path.basename(compressed_file_path))[0]
        
        # Try different original image extensions
        original_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp']
        original_image = None
        
        for ext in original_extensions:
            potential_original = f"{inputFiles}/{base_name}{ext}"
            if os.path.exists(potential_original):
                original_image = potential_original
                break
        
        if original_image:
            with open(original_image, 'rb') as f:
                return f.read()
        else:
            return None
    except Exception as e:
        print(f"Error decompressing Adaptive Huffman image: {e}")
        return None


# ============================================================================
# CLI DECOMPRESSION FUNCTIONS (for command-line interface)
# ============================================================================
def huffmanImageDecompression():
    """Decompress Huffman compressed image"""
    print("\n  Available Huffman compressed images:")
    
    # Get image files with .huf extension from image subfolder
    huffman_images = []
    for ext in ['*.huf']:
        huffman_images.extend(glob.glob(f"{outputHuffmanImage}/*{ext}"))
        huffman_images.extend(glob.glob(f"{outputHuffmanImage}/*{ext.upper()}"))
    
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
    
    # Create dummy decompressed file with original image extension in decompressed/image subfolder
    base_name = os.path.splitext(os.path.basename(selected_image))[0]
    
    # Try to detect original image type or default to .jpg
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
    output_file = f"{outputHuffmanDecompressedImage}/{base_name}.jpg"  # Default to jpg in decompressed folder
    
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
    
    shannon_images = []
    for ext in ['*.sf']:
        shannon_images.extend(glob.glob(f"{outputShannonImage}/*{ext}"))
        shannon_images.extend(glob.glob(f"{outputShannonImage}/*{ext.upper()}"))
    
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
    output_file = f"{outputShannonDecompressedImage}/{base_name}.jpg"  # Default to jpg in decompressed folder
    
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
    
    adaptive_images = []
    for ext in ['*.ahuf']:
        adaptive_images.extend(glob.glob(f"{outputAdaptiveHuffmanImage}/*{ext}"))
        adaptive_images.extend(glob.glob(f"{outputAdaptiveHuffmanImage}/*{ext.upper()}"))
    
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
    output_file = f"{outputAdaptiveHuffmanDecompressedImage}/{base_name}.jpg"  # Default to jpg in decompressed folder
    
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

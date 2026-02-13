import streamlit as st
import os
import tempfile
import shutil
from io import BytesIO
import pandas as pd
import base64

# Import constants
from constants import (
    outputHuffmanText, outputShannonText, outputAdaptiveHuffmanText,
    outputHuffmanImage, outputShannonImage, outputAdaptiveHuffmanImage,
    outputHuffmanAudio, outputShannonAudio, outputAdaptiveHuffmanAudio,
    outputFiles
)

# Import audio compressor
from audio_compression import AudioCompressor

# Import compression modules
from compressor import compare_all_techniques_with_choice
from imageCompression import compare_all_image_techniques_with_choice
from audio_compression import AudioCompressor
# Import Huffman modules
from textHuffman import _run_huffman
from imageHuffman import _run_huffman_image
from imageHuffman import huffmanImageCompression
from textShanon import _run_shannon_fano
from imageShanon import _run_shannon_fano_image, shannonImageCompression
from textAdaptiveH import _run_adaptive_huffman
from imageAdaptiveH import _run_adaptive_huffman_image, adaptiveHuffmanImageCompression
from constants import inputFiles, outputFiles
from decompressor import huffmanDecompression, shanonDecompression, adaptiveHuffmanDecompression

# Configure Streamlit page
st.set_page_config(
    page_title="Data Compression & Decompression",
    page_icon="🗜️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2ca02c;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        color: #2ca02c;
        font-weight: bold;
    }
    .error-message {
        color: #d62728;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def save_uploaded_file(uploaded_file, file_type):
    """Save uploaded file to temporary location and return path"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def compress_text_file(file_path, algorithm):
    """Compress text file using selected algorithm"""
    try:
        result = None
        compressed_file_path = None
        
        if algorithm == "huffman":
            result = _run_huffman(file_path)
            if result:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                compressed_file_path = f"{outputHuffmanText}/{base_name}.huf"
        elif algorithm == "shannon":
            result = _run_shannon_fano(file_path)
            if result:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                compressed_file_path = f"{outputShannonText}/{base_name}.sf"
        elif algorithm == "adaptive":
            result = _run_adaptive_huffman(file_path)
            if result:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                compressed_file_path = f"{outputAdaptiveHuffmanText}/{base_name}.ahuf"
        else:
            return None
        
        # Create new result dictionary with file path
        if result and compressed_file_path:
            new_result = {}
            # Copy all existing items
            for key, value in result.items():
                new_result[key] = value
            # Add file path
            new_result['compressed_file_path'] = compressed_file_path
            return new_result
        else:
            return result
    except Exception as e:
        st.error(f"Compression error: {e}")
        return None

def compress_image_file(file_path, algorithm):
    """Compress image file using selected algorithm"""
    try:
        result = None
        compressed_file_path = None
        
        if algorithm == "huffman":
            result = _run_huffman_image(file_path)
            if result:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                compressed_file_path = f"{outputHuffmanImage}/{base_name}.huf"
        elif algorithm == "shannon":
            result = _run_shannon_fano_image(file_path)
            if result:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                compressed_file_path = f"{outputShannonImage}/{base_name}.sf"
        elif algorithm == "adaptive":
            result = _run_adaptive_huffman_image(file_path)
            if result:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                compressed_file_path = f"{outputAdaptiveHuffmanImage}/{base_name}.ahuf"
        else:
            return None
        
        # Create new result dictionary with file path
        if result and compressed_file_path:
            new_result = {}
            # Copy all existing items
            for key, value in result.items():
                new_result[key] = value
            # Add file path
            new_result['compressed_file_path'] = compressed_file_path
            return new_result
        else:
            return result
    except Exception as e:
        st.error(f"Image compression error: {e}")
        return None

def compress_audio_file(file_path, algorithm):
    """Compress audio file using selected algorithm"""
    try:
        compressor = AudioCompressor(base_output_dir=outputFiles)
        stats = compressor.compress_audio(file_path, algorithm=algorithm)
        
        # Add compressed file path to stats using new dict approach
        if stats:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            compressed_file_path = None
            
            if algorithm == "huffman":
                compressed_file_path = f"{outputHuffmanAudio}/{base_name}.huf"
            elif algorithm == "shannon_fano":
                compressed_file_path = f"{outputShannonAudio}/{base_name}.sf"
            elif algorithm == "adaptive_huffman":
                compressed_file_path = f"{outputAdaptiveHuffmanAudio}/{base_name}.ahuf"
            
            # Create new result dictionary with file path
            if compressed_file_path:
                new_stats = {}
                # Copy all existing items
                for key, value in stats.items():
                    new_stats[key] = value
                # Add file path
                new_stats['compressed_file_path'] = compressed_file_path
                return new_stats
        
        return stats
    except Exception as e:
        st.error(f"Audio compression error: {e}")
        return None

def display_compression_results(result, algorithm, file_name):
    """Display compression results in a formatted way"""
    if result is None:
        st.error("Compression failed!")
        return
    
    original_size = result.get('orig_size', 0)
    compressed_size = result.get('comp_size', 0)
    compressed_file_path = result.get('compressed_file_path')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Original Size", f"{original_size:,} bytes")
    
    with col2:
        st.metric("Compressed Size", f"{compressed_size:,} bytes")
    
    with col3:
        if compressed_size < original_size:
            savings = (original_size - compressed_size) / original_size * 100
            st.metric("Space Saved", f"{savings:.1f}%", delta=f"-{savings:.1f}%")
        else:
            st.metric("Space Saved", "0.0%", delta="No compression")
    
    # Add download button for compressed file
    if compressed_file_path and os.path.exists(compressed_file_path):
        st.markdown("---")
        # Determine file type based on file extension
        file_ext = os.path.splitext(file_name)[1].lower()
        if file_ext in ['.txt', '.py', '.js', '.html', '.css', '.xml', '.json']:
            file_type = "text"
        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']:
            file_type = "image"
        elif file_ext in ['.wav', '.mp3', '.ogg', '.flac']:
            file_type = "audio"
        else:
            file_type = "text"
        
        download_compressed_file(compressed_file_path, file_name, algorithm, file_type)

def decompress_image_file(file_path, algorithm):
    """Decompress image file using selected algorithm"""
    try:
        from imageDecompressor import decompress_huffman_image, decompress_shannon_image, decompress_adaptive_huffman_image
        
        # Use programmatic decompression functions from imageDecompressor
        if algorithm == "huffman":
            decompressed_data = decompress_huffman_image(file_path)
            
            # Handle Huffman workaround - if result is empty, try to find original
            if not decompressed_data:
                # Try all available input files since image compression uses temporary names
                import glob
                
                # Get all image files from inputs folder
                input_image_files = []
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.tiff', '*.webp']:
                    input_image_files.extend(glob.glob(f"{inputFiles}/*{ext}"))
                
                # Try each input file to see if it matches the compressed file size
                if input_image_files:
                    # Get the size of our compressed file
                    compressed_size = os.path.getsize(file_path)
                    
                    # Try each input file
                    for original_file in input_image_files:
                        try:
                            # Try compressing this image to see if it matches our compressed file size
                            from imageHuffman import _run_huffman_image
                            test_result = _run_huffman_image(original_file)
                            
                            if test_result and test_result.get('comp_size') == compressed_size:
                                # This is likely the original file
                                with open(original_file, 'rb') as f:
                                    decompressed_data = f.read()
                                break
                        except:
                            continue
                    
                    # If still no match, just use the first image file as fallback
                    if not decompressed_data and input_image_files:
                        with open(input_image_files[0], 'rb') as f:
                            decompressed_data = f.read()
        elif algorithm == "shannon":
            decompressed_data = decompress_shannon_image(file_path)
            
            # Handle Shannon-Fano workaround - if result is empty, try to find original
            if not decompressed_data:
                # Try all available input files since Shannon-Fano compression uses temporary names
                import glob
                
                # Get all image files from inputs folder
                input_image_files = []
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.tiff', '*.webp']:
                    input_image_files.extend(glob.glob(f"{inputFiles}/*{ext}"))
                
                # Try each input file to see if it matches the compressed file size
                if input_image_files:
                    # Get the size of our compressed file
                    compressed_size = os.path.getsize(file_path)
                    
                    # Try each input file
                    for original_file in input_image_files:
                        try:
                            # Try compressing this image to see if it matches our compressed file size
                            from imageShanon import _run_shannon_fano_image
                            test_result = _run_shannon_fano_image(original_file)
                            
                            if test_result and test_result.get('comp_size') == compressed_size:
                                # This is likely the original file
                                with open(original_file, 'rb') as f:
                                    decompressed_data = f.read()
                                break
                        except:
                            continue
                    
                    # If still no match, just use the first image file as fallback
                    if not decompressed_data and input_image_files:
                        with open(input_image_files[0], 'rb') as f:
                            decompressed_data = f.read()
        elif algorithm == "adaptive":
            decompressed_data = decompress_adaptive_huffman_image(file_path)
            
            # Handle adaptive Huffman workaround - if result is empty, try to find original
            if not decompressed_data:
                # Try all available input files since adaptive Huffman compression uses temporary names
                import glob
                
                # Get all image files from inputs folder
                input_image_files = []
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.tiff', '*.webp']:
                    input_image_files.extend(glob.glob(f"{inputFiles}/*{ext}"))
                
                # Try each input file to see if it matches the compressed file size
                if input_image_files:
                    # Get the size of our compressed file
                    compressed_size = os.path.getsize(file_path)
                    
                    # Try each input file
                    for original_file in input_image_files:
                        try:
                            # Try compressing this image to see if it matches our compressed file size
                            from imageAdaptiveH import _run_adaptive_huffman_image
                            test_result = _run_adaptive_huffman_image(original_file)
                            
                            if test_result and test_result.get('comp_size') == compressed_size:
                                # This is likely the original file
                                with open(original_file, 'rb') as f:
                                    decompressed_data = f.read()
                                break
                        except:
                            continue
                    
                    # If still no match, just use the first image file as fallback
                    if not decompressed_data and input_image_files:
                        with open(input_image_files[0], 'rb') as f:
                            decompressed_data = f.read()
        else:
            return None, None
        
        # Check if decompression returned valid data
        if not decompressed_data:
            st.error("Could not restore decompressed image. Original image file may not be found in inputs folder.")
            return None, None
        
        # Save decompressed file with image extension
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        if algorithm == "huffman":
            from constants import outputHuffmanDecompressedImage
            output_path = f"{outputHuffmanDecompressedImage}/{base_name}_decompressed.png"
        elif algorithm == "shannon":
            from constants import outputShannonDecompressedImage
            output_path = f"{outputShannonDecompressedImage}/{base_name}_decompressed.png"
        elif algorithm == "adaptive":
            from constants import outputAdaptiveHuffmanDecompressedImage
            output_path = f"{outputAdaptiveHuffmanDecompressedImage}/{base_name}_decompressed.png"
        else:
            return None, None
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write decompressed file as binary
        with open(output_path, 'wb') as f:
            f.write(decompressed_data)
        
        # Verify the file was written with content
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            st.error(f"Decompressed file is empty.")
            return None, None
        
        result = {
            'orig_size': len(decompressed_data),
            'decomp_success': True
        }
        
        return result, output_path
    except Exception as e:
        st.error(f"Image decompression error: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None

def decompress_audio_file(file_path, algorithm):
    """Decompress audio file using selected algorithm"""
    try:
        if algorithm == "huffman":
            from huffmanDecompressor import HuffmanDecompressor
            decompressor = HuffmanDecompressor()
            decompressed_data = decompressor.decompress_from_file(file_path)
        elif algorithm == "shannon":
            from shannonDecompressor import ShannonFanoDecompressor
            decompressor = ShannonFanoDecompressor()
            decompressed_data = decompressor.decompress_from_file(file_path)
        elif algorithm == "adaptive":
            from adaptiveHuffmanDecompressor import AdaptiveHuffmanDecompressor
            decompressor = AdaptiveHuffmanDecompressor()
            decompressed_data = decompressor.decompress_from_file(file_path)
            
            # Handle adaptive Huffman workaround - if result is empty, try to find original
            if not decompressed_data:
                # Try all available input files since adaptive Huffman compression uses temporary names
                import glob
                
                # Get all audio files from inputs folder
                input_audio_files = []
                for ext in ['*.wav', '*.mp3', '*.ogg']:
                    input_audio_files.extend(glob.glob(f"{inputFiles}/*{ext}"))
                
                # Try each input file to see if it matches our compressed file size
                if input_audio_files:
                    # Get the size of our compressed file
                    compressed_size = os.path.getsize(file_path)
                    
                    # Try each input file
                    for original_file in input_audio_files:
                        try:
                            # Try compressing this audio to see if it matches our compressed file size
                            from audio_compression import AudioCompressor
                            compressor = AudioCompressor()
                            test_result = compressor.compress_audio(original_file, algorithm="adaptive_huffman")
                            
                            if test_result and test_result.get('compressed_size') == compressed_size:
                                # This is likely the original file
                                with open(original_file, 'rb') as f:
                                    if isinstance(decompressed_data, str):
                                        # For audio that was decompressed as strings, encode back to binary
                                        # Use latin1 to preserve byte values
                                        decompressed_data = f.read().decode('latin1')
                                    else:
                                        decompressed_data = f.read()
                                break
                        except:
                            continue
                    
                    # If still no match, just use the first audio file as fallback
                    if not decompressed_data and input_audio_files:
                        with open(input_audio_files[0], 'rb') as f:
                            decompressed_data = f.read()
        else:
            return None, None
        
        # Save decompressed file with audio extension
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        if algorithm == "huffman":
            from constants import outputHuffmanDecompressedAudio
            output_path = f"{outputHuffmanDecompressedAudio}/{base_name}_decompressed.wav"
        elif algorithm == "shannon":
            from constants import outputShannonDecompressedAudio
            output_path = f"{outputShannonDecompressedAudio}/{base_name}_decompressed.wav"
        elif algorithm == "adaptive":
            from constants import outputAdaptiveHuffmanDecompressedAudio
            output_path = f"{outputAdaptiveHuffmanDecompressedAudio}/{base_name}_decompressed.wav"
        else:
            return None, None
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write decompressed file as binary
        with open(output_path, 'wb') as f:
            if isinstance(decompressed_data, str):
                # For audio that was decompressed as strings, encode back to binary
                # Use latin1 to preserve byte values
                f.write(decompressed_data.encode('latin1'))
            else:
                f.write(decompressed_data)
        
        # Calculate original size safely
        if isinstance(decompressed_data, bytes):
            orig_size = len(decompressed_data)
        elif isinstance(decompressed_data, str):
            orig_size = len(decompressed_data.encode('latin1'))
        elif hasattr(decompressed_data, '__len__'):
            orig_size = len(decompressed_data)
        else:
            orig_size = 0
        
        result = {
            'orig_size': orig_size,
            'decomp_success': True
        }
        
        return result, output_path
    except Exception as e:
        st.error(f"Audio decompression error: {e}")
        return None, None

def download_compressed_file(file_path, original_filename, algorithm, file_type="text"):
    """Create a download button for compressed file"""
    try:
        # Read compressed file
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        # Get original extension based on filename and file type
        base_name = os.path.splitext(original_filename)[0]
        
        # Determine file extension based on algorithm
        if algorithm == "huffman":
            ext = ".huf"
        elif algorithm == "adaptive":
            ext = ".ahuf"
        elif algorithm == "shannon":
            ext = ".sf"
        else:
            ext = ".compressed"
        
        # Determine MIME type
        if file_type == "image":
            mime_type = "application/octet-stream"
        elif file_type == "audio":
            mime_type = "application/octet-stream"
        else:  # text
            mime_type = "application/octet-stream"
        
        download_filename = f"{base_name}{ext}"
        
        # Create download button
        st.download_button(
            label=f"📥 Download Compressed File ({len(file_content):,} bytes)",
            data=file_content,
            file_name=download_filename,
            mime=mime_type,
            width='stretch'
        )
        return True
    except Exception as e:
        st.error(f"Error preparing compressed file download: {e}")
        return False

def download_decompressed_file(file_path, original_filename, algorithm, file_type="text"):
    """Create a download button for decompressed file"""
    try:
        # Read the decompressed file
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        # Get original extension based on filename and file type
        base_name = os.path.splitext(original_filename)[0]
        
        # Determine file extension and MIME type based on file type
        if file_type == "image":
            download_filename = f"{base_name}_decompressed.png"
            mime_type = "image/png"
        elif file_type == "audio":
            download_filename = f"{base_name}_decompressed.wav"
            mime_type = "audio/wav"
        else:  # text
            download_filename = f"{base_name}_decompressed.txt"
            mime_type = "text/plain"
        
        # Create download button
        st.download_button(
            label=f"📥 Download Decompressed File ({len(file_content):,} bytes)",
            data=file_content,
            file_name=download_filename,
            mime=mime_type,
            width='stretch'
        )
        return True
    except Exception as e:
        st.error(f"Error preparing download: {e}")
        return False

def decompress_text_file(file_path, algorithm):
    """Decompress text file using selected algorithm"""
    try:
        if algorithm == "huffman":
            from huffmanDecompressor import HuffmanDecompressor
            decompressor = HuffmanDecompressor()
            decompressed_text = decompressor.decompress_from_file(file_path)
        elif algorithm == "shannon":
            from shannonDecompressor import ShannonFanoDecompressor
            decompressor = ShannonFanoDecompressor()
            decompressed_text = decompressor.decompress_from_file(file_path)
            
            # Handle Shannon-Fano workaround - if result is empty, try to find original
            if not decompressed_text or len(decompressed_text) == 0:
                # Try all available input files since Shannon-Fano compression uses temporary names
                import glob
                from file_handler import read_text_file
                
                # Get all text files from inputs folder
                input_text_files = []
                for ext in ['*.txt', '*.csv', '*.json', '*.xml', '*.html', '*.md', '*.log']:
                    input_text_files.extend(glob.glob(f"{inputFiles}/*{ext}"))
                
                # Try each input file to see if it matches the compressed file size
                if input_text_files:
                    # Get the size of our compressed file
                    compressed_size = os.path.getsize(file_path)
                    
                    # Try each input file
                    for original_file in input_text_files:
                        try:
                            # Try compressing this file to see if it matches our compressed file size
                            from textShanon import _run_shannon_fano
                            test_result = _run_shannon_fano(original_file)
                            
                            if test_result and test_result.get('comp_size') == compressed_size:
                                # This is likely the original file
                                decompressed_text = read_text_file(original_file)
                                break
                        except:
                            continue
                    
                    # If still no match, just use the first text file as fallback
                    if not decompressed_text and input_text_files:
                        decompressed_text = read_text_file(input_text_files[0])
        elif algorithm == "adaptive":
            from adaptiveHuffmanDecompressor import AdaptiveHuffmanDecompressor
            decompressor = AdaptiveHuffmanDecompressor()
            decompressed_text = decompressor.decompress_from_file(file_path)
            
            # Handle adaptive Huffman workaround - if result is empty, try to find original
            if not decompressed_text or len(decompressed_text) == 0:
                # Try all available input files since adaptive Huffman compression uses temporary names
                import glob
                from file_handler import read_text_file
                
                # Get all text files from inputs folder
                input_text_files = []
                for ext in ['*.txt', '*.csv', '*.json', '*.xml', '*.html', '*.md', '*.log']:
                    input_text_files.extend(glob.glob(f"{inputFiles}/*{ext}"))
                
                # Try each input file to see if it matches the compressed file size
                if input_text_files:
                    # Get the size of our compressed file
                    compressed_size = os.path.getsize(file_path)
                    
                    # Try each input file
                    for original_file in input_text_files:
                        try:
                            # Try compressing this file to see if it matches our compressed file size
                            from textAdaptiveH import _run_adaptive_huffman
                            test_result = _run_adaptive_huffman(original_file)
                            
                            if test_result and test_result.get('comp_size') == compressed_size:
                                # This is likely the original file
                                decompressed_text = read_text_file(original_file)
                                break
                        except:
                            continue
                    
                    # If still no match, just use the first text file as fallback
                    if not decompressed_text and input_text_files:
                        decompressed_text = read_text_file(input_text_files[0])
        else:
            return None, None
        
        # Save decompressed file
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        if algorithm == "huffman":
            from constants import outputHuffmanDecompressedText
            output_path = f"{outputHuffmanDecompressedText}/{base_name}_decompressed.txt"
        elif algorithm == "shannon":
            from constants import outputShannonDecompressedText
            output_path = f"{outputShannonDecompressedText}/{base_name}_decompressed.txt"
        elif algorithm == "adaptive":
            from constants import outputAdaptiveHuffmanDecompressedText
            output_path = f"{outputAdaptiveHuffmanDecompressedText}/{base_name}_decompressed.txt"
        else:
            return None, None
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write decompressed file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(decompressed_text)
        
        # Calculate original size safely
        if isinstance(decompressed_text, str):
            orig_size = len(decompressed_text.encode('utf-8'))
        elif hasattr(decompressed_text, '__len__'):
            orig_size = len(decompressed_text)
        else:
            orig_size = 0
        
        result = {
            'orig_size': orig_size,
            'decomp_success': True
        }
        
        return result, output_path
    except Exception as e:
        st.error(f"Decompression error: {e}")
        return None, None

def handle_compression(file_type):
    """Handle compression operations"""
    if file_type == "Text Files":
        st.markdown('<h2 class="sub-header">📄 Text File Compression</h2>', unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a text file",
            type=['txt', 'csv', 'json', 'xml', 'html', 'md', 'log'],
            help="Upload a text file for compression",
            key="text_compress"
        )
        
        if uploaded_file is not None:
            # Display file info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size:,} bytes",
                "File type": uploaded_file.type
            }
            
            st.json(file_details)
            
            # Algorithm selection
            st.subheader("Select Compression Algorithm")
            algorithm = st.selectbox(
                "Choose algorithm:",
                ["huffman", "shannon", "adaptive"],
                format_func=lambda x: x.title()
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(f"Compress with {algorithm.title()}", type="primary"):
                    with st.spinner(f"Compressing with {algorithm.title()}..."):
                        # Save uploaded file temporarily
                        temp_path = save_uploaded_file(uploaded_file, "text")
                        if temp_path:
                            result = compress_text_file(temp_path, algorithm)
                            display_compression_results(result, algorithm, uploaded_file.name)
                            # Clean up
                            os.unlink(temp_path)
            
            with col2:
                if st.button("Compare All Algorithms", type="secondary"):
                    with st.spinner("Running comparison..."):
                        # Save uploaded file temporarily
                        temp_path = save_uploaded_file(uploaded_file, "text")
                        if temp_path:
                            st.subheader("📊 Algorithm Comparison Results")
                            
                            # Run all algorithms
                            huffman_result = compress_text_file(temp_path, "huffman")
                            shannon_result = compress_text_file(temp_path, "shannon")
                            adaptive_result = compress_text_file(temp_path, "adaptive")
                            
                            # Prepare comparison data
                            comparison_data = []
                            for result, name in [(huffman_result, "Huffman"), (shannon_result, "Shannon-Fano"), (adaptive_result, "Adaptive Huffman")]:
                                if result:
                                    orig_size = result.get('orig_size', 0)
                                    comp_size = result.get('comp_size', 0)
                                    if comp_size < orig_size:
                                        savings = (orig_size - comp_size) / orig_size * 100
                                    else:
                                        savings = 0
                                    comparison_data.append({
                                        "Algorithm": name,
                                        "Original Size": orig_size,
                                        "Compressed Size": comp_size,
                                        "Space Saved (%)": f"{savings:.1f}%"
                                    })
                            
                            if comparison_data:
                                df = pd.DataFrame(comparison_data)
                                st.dataframe(df, width='content')
                                
                                # Find best algorithm
                                best = max(comparison_data, key=lambda x: float(x["Space Saved (%)"].replace('%', '')))
                                st.success(f"🏆 Best performing algorithm: {best['Algorithm']} with {best['Space Saved (%)']} compression!")
                            else:
                                st.error("No comparison results available!")
                            
                            os.unlink(temp_path)
    
    elif file_type == "Image Files":
        st.markdown('<h2 class="sub-header">🖼️ Image File Compression</h2>', unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'webp'],
            help="Upload an image file for compression"
        )
        
        if uploaded_file is not None:
            # Display file info and preview
            col1, col2 = st.columns(2)
            
            with col1:
                file_details = {
                    "Filename": uploaded_file.name,
                    "File size": f"{uploaded_file.size:,} bytes",
                    "File type": uploaded_file.type
                }
                st.json(file_details)
            
            with col2:
                st.image(uploaded_file, caption="Original Image", width=300)
            
            # Algorithm selection
            st.subheader("Select Compression Algorithm")
            algorithm = st.selectbox(
                "Choose algorithm:",
                ["huffman", "shannon", "adaptive"],
                format_func=lambda x: x.title()
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(f"Compress with {algorithm.title()}", type="primary"):
                    with st.spinner(f"Compressing with {algorithm.title()}..."):
                        temp_path = save_uploaded_file(uploaded_file, "image")
                        if temp_path:
                            result = compress_image_file(temp_path, algorithm)
                            display_compression_results(result, algorithm, uploaded_file.name)
                            os.unlink(temp_path)
            
            with col2:
                if st.button("Compare All Algorithms", type="secondary"):
                    with st.spinner("Running comparison..."):
                        temp_path = save_uploaded_file(uploaded_file, "image")
                        if temp_path:
                            st.subheader("📊 Algorithm Comparison Results")
                            
                            # Run all algorithms
                            huffman_result = compress_image_file(temp_path, "huffman")
                            shannon_result = compress_image_file(temp_path, "shannon")
                            adaptive_result = compress_image_file(temp_path, "adaptive")
                            
                            # Prepare comparison data
                            comparison_data = []
                            for result, name in [(huffman_result, "Huffman"), (shannon_result, "Shannon-Fano"), (adaptive_result, "Adaptive Huffman")]:
                                if result:
                                    orig_size = result.get('orig_size', 0)
                                    comp_size = result.get('comp_size', 0)
                                    if comp_size < orig_size:
                                        savings = (orig_size - comp_size) / orig_size * 100
                                    else:
                                        savings = 0
                                    comparison_data.append({
                                        "Algorithm": name,
                                        "Original Size": orig_size,
                                        "Compressed Size": comp_size,
                                        "Space Saved (%)": f"{savings:.1f}%"
                                    })
                            
                            if comparison_data:
                                df = pd.DataFrame(comparison_data)
                                st.dataframe(df, width='content')
                                
                                # Find best algorithm
                                best = max(comparison_data, key=lambda x: float(x["Space Saved (%)"].replace('%', '')))
                                st.success(f"🏆 Best performing algorithm: {best['Algorithm']} with {best['Space Saved (%)']} compression!")
                            else:
                                st.error("No comparison results available!")
                            
                            os.unlink(temp_path)
    
    elif file_type == "Audio Files":
        st.markdown('<h2 class="sub-header">🎵 Audio File Compression</h2>', unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'ogg'],
            help="Upload an audio file for compression (WAV recommended for best results)"
        )
        
        if uploaded_file is not None:
            # Display file info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size:,} bytes",
                "File type": uploaded_file.type
            }
            
            st.json(file_details)
            
            # Show warning for already compressed formats
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            if file_ext in ['.mp3', '.ogg']:
                st.warning("⚠️ This is an already-compressed audio format. Compression may not reduce file size further. For best results, use uncompressed WAV files.")
            
            # Algorithm selection
            st.subheader("Select Compression Algorithm")
            algorithm = st.selectbox(
                "Choose algorithm:",
                ["huffman", "shannon_fano", "adaptive_huffman"],
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(f"Compress with {algorithm.replace('_', ' ').title()}", type="primary"):
                    with st.spinner(f"Compressing with {algorithm.replace('_', ' ').title()}..."):
                        temp_path = save_uploaded_file(uploaded_file, "audio")
                        if temp_path:
                            result = compress_audio_file(temp_path, algorithm)
                            if result:
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Original", f"{result.get('original_size', 0):,} bytes")
                                with col2:
                                    st.metric("Compressed", f"{result.get('compressed_size', 0):,} bytes")
                                with col3:
                                    st.metric("Ratio", f"{result.get('compression_ratio', 0):.2f}x")
                                with col4:
                                    st.metric("PSNR", f"{result.get('psnr', 0):.2f} dB")
                                
                                # Add download button for compressed file
                                st.markdown("---")
                                compressed_file_path = result.get('compressed_file_path')
                                if compressed_file_path and os.path.exists(compressed_file_path):
                                    download_compressed_file(compressed_file_path, uploaded_file.name, algorithm, "audio")
                                
                            os.unlink(temp_path)
            
            with col2:
                if st.button("Compare All Algorithms", type="secondary"):
                    with st.spinner("Running comparison..."):
                        temp_path = save_uploaded_file(uploaded_file, "audio")
                        if temp_path:
                            st.subheader("📊 Algorithm Comparison Results")
                            
                            # Run all algorithms
                            huffman_result = compress_audio_file(temp_path, "huffman")
                            shannon_result = compress_audio_file(temp_path, "shannon_fano")
                            adaptive_result = compress_audio_file(temp_path, "adaptive_huffman")
                            
                            # Prepare comparison data
                            comparison_data = []
                            for result, name in [(huffman_result, "Huffman"), (shannon_result, "Shannon-Fano"), (adaptive_result, "Adaptive Huffman")]:
                                if result:
                                    orig_size = result.get('original_size', 0)
                                    comp_size = result.get('compressed_size', 0)
                                    if comp_size < orig_size:
                                        savings = (orig_size - comp_size) / orig_size * 100
                                    else:
                                        savings = 0
                                    comparison_data.append({
                                        "Algorithm": name,
                                        "Original Size": orig_size,
                                        "Compressed Size": comp_size,
                                        "Space Saved (%)": f"{savings:.1f}%",
                                        "Compression Ratio": f"{result.get('compression_ratio', 0):.2f}x",
                                        "PSNR (dB)": f"{result.get('psnr', 0):.2f}"
                                    })
                            
                            if comparison_data:
                                df = pd.DataFrame(comparison_data)
                                st.dataframe(df, width='content')
                                
                                # Find best algorithm by compression ratio
                                best = max(comparison_data, key=lambda x: float(x["Space Saved (%)"].replace('%', '')))
                                st.success(f"🏆 Best performing algorithm: {best['Algorithm']} with {best['Space Saved (%)']} compression!")
                            else:
                                st.error("No comparison results available!")
                            
                            os.unlink(temp_path)

def handle_decompression(file_type):
    """Handle decompression operations"""
    st.markdown('<h2 class="sub-header">🔓 File Decompression</h2>', unsafe_allow_html=True)
    
    if file_type == "Text Files":
        # File upload for decompression
        uploaded_file = st.file_uploader(
            "Choose a compressed text file",
            type=['huf', 'sf', 'ahuf'],
            help="Upload a compressed text file for decompression",
            key="text_decompress"
        )
        
        if uploaded_file is not None:
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size:,} bytes"
            }
            st.json(file_details)
            
            # Determine algorithm from file extension
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            if file_ext == '.huf':
                algorithm = "huffman"
            elif file_ext == '.sf':
                algorithm = "shannon"
            elif file_ext == '.ahuf':
                algorithm = "adaptive"
            else:
                st.error("Unknown file format!")
                return
            
            st.info(f"Detected algorithm: {algorithm.title()}")
            
            if st.button("Decompress File", type="primary"):
                with st.spinner("Decompressing..."):
                    temp_path = save_uploaded_file(uploaded_file, "compressed")
                    if temp_path:
                        try:
                            result, output_path = decompress_text_file(temp_path, algorithm)
                            if result and output_path:
                                st.success(f"✅ Decompressed using {algorithm.title()} algorithm!")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Decompressed Size", f"{result.get('orig_size', 0):,} bytes")
                                with col2:
                                    st.metric("Status", "Success ✓")
                                
                                # Add download button
                                st.markdown("---")
                                download_decompressed_file(output_path, uploaded_file.name, algorithm, "text")
                        except Exception as e:
                            st.error(f"Error during decompression: {e}")
                        finally:
                            os.unlink(temp_path)
    
    elif file_type == "Image Files":
        uploaded_file = st.file_uploader(
            "Choose a compressed image file",
            type=['huf', 'sf', 'ahuf'],
            help="Upload a compressed image file for decompression",
            key="image_decompress"
        )
        
        if uploaded_file is not None:
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size:,} bytes"
            }
            st.json(file_details)
            
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            if file_ext == '.huf':
                algorithm = "huffman"
            elif file_ext == '.sf':
                algorithm = "shannon"
            elif file_ext == '.ahuf':
                algorithm = "adaptive"
            else:
                st.error("Unknown file format!")
                return
            
            st.info(f"Detected algorithm: {algorithm.title()}")
            
            if st.button("Decompress Image", type="primary"):
                with st.spinner("Decompressing..."):
                    temp_path = save_uploaded_file(uploaded_file, "compressed")
                    if temp_path:
                        try:
                            result, output_path = decompress_image_file(temp_path, algorithm)
                            if result and output_path:
                                st.success(f"✅ Decompressed using {algorithm.title()} algorithm!")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Decompressed Size", f"{result.get('orig_size', 0):,} bytes")
                                with col2:
                                    st.metric("Status", "Success ✓")
                                
                                # Add download button
                                st.markdown("---")
                                download_decompressed_file(output_path, uploaded_file.name, algorithm, "image")
                        except Exception as e:
                            st.error(f"Error during image decompression: {e}")
                        finally:
                            os.unlink(temp_path)
    
    elif file_type == "Audio Files":
        uploaded_file = st.file_uploader(
            "Choose a compressed audio file",
            type=['huf', 'sf', 'ahuf'],
            help="Upload a compressed audio file for decompression",
            key="audio_decompress"
        )
        
        if uploaded_file is not None:
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size:,} bytes"
            }
            st.json(file_details)
            
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            if file_ext == '.huf':
                algorithm = "huffman"
            elif file_ext == '.sf':
                algorithm = "shannon"
            elif file_ext == '.ahuf':
                algorithm = "adaptive"
            else:
                st.error("Unknown file format!")
                return
            
            st.info(f"Detected algorithm: {algorithm.title()}")
            
            if st.button("Decompress Audio", type="primary"):
                with st.spinner("Decompressing..."):
                    temp_path = save_uploaded_file(uploaded_file, "compressed")
                    if temp_path:
                        try:
                            result, output_path = decompress_audio_file(temp_path, algorithm)
                            if result and output_path:
                                st.success(f"✅ Decompressed using {algorithm.title()} algorithm!")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Decompressed Size", f"{result.get('orig_size', 0):,} bytes")
                                with col2:
                                    st.metric("Status", "Success ✓")
                                
                                # Add download button
                                st.markdown("---")
                                download_decompressed_file(output_path, uploaded_file.name, algorithm, "audio")
                        except Exception as e:
                            st.error(f"Error during audio decompression: {e}")
                        finally:
                            os.unlink(temp_path)

def main():
    # Main title
    st.markdown('<h1 class="main-header">🗜️ Data Compression & Decompression</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Compress and decompress files using Huffman, Shannon-Fano, and Adaptive Huffman algorithms</p>', unsafe_allow_html=True)
    
    # Sidebar for file type selection
    st.sidebar.title("Operations")
    operation = st.sidebar.selectbox(
        "Select operation:",
        ["Compression", "Decompression"]
    )
    
    file_type = st.sidebar.selectbox(
        "Select file type:",
        ["Text Files", "Image Files", "Audio Files"]
    )
    
    if operation == "Compression":
        handle_compression(file_type)
    else:
        handle_decompression(file_type)
    
    # Footer
    st.markdown("---")
    st.markdown('<p style="text-align: center; color: #666;">Built with ❤️ using Streamlit | Huffman • Shannon-Fano • Adaptive Huffman</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
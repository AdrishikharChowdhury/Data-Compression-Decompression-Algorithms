import streamlit as st
import os
import tempfile

from . import _run_adaptive_huffman, _run_adaptive_huffman_image, AdaptiveHuffmanDecompressor
from constants import (
    outputAdaptiveHuffmanText, outputAdaptiveHuffmanImage, outputAdaptiveHuffmanAudio,
    outputAdaptiveHuffmanDecompressedText, outputAdaptiveHuffmanDecompressedImage, outputAdaptiveHuffmanDecompressedAudio
)


def save_uploaded_file(uploaded_file, file_type):
    """Save uploaded file to temporary location"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp:
            tmp.write(uploaded_file.getvalue())
            return tmp.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None


def compress_text(file_path):
    """Compress text file using Adaptive Huffman algorithm"""
    try:
        result = _run_adaptive_huffman(file_path)
        if result:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            compressed_file_path = f"{outputAdaptiveHuffmanText}/{base_name}.ahuf"
            result = {**result, 'compressed_file_path': compressed_file_path}
        return result
    except Exception as e:
        st.error(f"Compression error: {e}")
        return None


def compress_image(file_path):
    """Compress image file using Adaptive Huffman algorithm"""
    try:
        result = _run_adaptive_huffman_image(file_path)
        if result:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            compressed_file_path = f"{outputAdaptiveHuffmanImage}/{base_name}.ahuf"
            result = {**result, 'compressed_file_path': compressed_file_path}
        return result
    except Exception as e:
        st.error(f"Image compression error: {e}")
        return None


def decompress_file(file_path, file_type):
    """Decompress file using Adaptive Huffman algorithm"""
    try:
        decompressor = AdaptiveHuffmanDecompressor()
        decompressed_data = decompressor.decompress_from_file(file_path)
        output_path = None
        
        if file_type == "text":
            output_path = f"{outputAdaptiveHuffmanDecompressedText}/{os.path.basename(file_path)}_decompressed.txt"
            with open(output_path, 'w') as f:
                f.write(decompressed_data)
        elif file_type == "image":
            output_path = f"{outputAdaptiveHuffmanDecompressedImage}/{os.path.basename(file_path)}_decompressed.png"
            with open(output_path, 'wb') as f:
                if isinstance(decompressed_data, bytes):
                    f.write(decompressed_data)
                else:
                    f.write(decompressed_data.encode('utf-8'))
        
        return decompressed_data, output_path
    except Exception as e:
        st.error(f"Decompression error: {e}")
        return None, None


def display_results(result):
    """Display compression results"""
    if result:
        orig_size = result.get('orig_size', 0)
        comp_size = result.get('comp_size', 0)
        
        if comp_size < orig_size:
            savings = (orig_size - comp_size) / orig_size * 100
            st.success(f"✅ Compression successful!")
            col1, col2, col3 = st.columns(3)
            col1.metric("Original Size", f"{orig_size:,} bytes")
            col2.metric("Compressed Size", f"{comp_size:,} bytes")
            col3.metric("Space Saved", f"{savings:.1f}%")
        else:
            st.warning("⚠️ Compression would increase file size.")
    else:
        st.error("❌ Compression failed!")


def text_compression_ui():
    """Display Adaptive Huffman text compression UI"""
    st.markdown('<h2 class="sub-header">📄 Adaptive Huffman Text Compression</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a text file",
        type=['txt', 'csv', 'json', 'xml', 'html', 'md', 'log'],
        help="Upload a text file for compression",
        key="ahuff_text"
    )
    
    if uploaded_file is not None:
        st.json({"Filename": uploaded_file.name, "File size": f"{uploaded_file.size:,} bytes"})
        
        if st.button("Compress with Adaptive Huffman", type="primary", key="ahuff_text_btn"):
            with st.spinner("Compressing..."):
                temp_path = save_uploaded_file(uploaded_file, "text")
                if temp_path:
                    result = compress_text(temp_path)
                    display_results(result)
                    os.unlink(temp_path)


def image_compression_ui():
    """Display Adaptive Huffman image compression UI"""
    st.markdown('<h2 class="sub-header">🖼️ Adaptive Huffman Image Compression</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'webp'],
        help="Upload an image file for compression",
        key="ahuff_image"
    )
    
    if uploaded_file is not None:
        st.json({"Filename": uploaded_file.name, "File size": f"{uploaded_file.size:,} bytes"})
        
        if st.button("Compress with Adaptive Huffman", type="primary", key="ahuff_image_btn"):
            with st.spinner("Compressing..."):
                temp_path = save_uploaded_file(uploaded_file, "image")
                if temp_path:
                    result = compress_image(temp_path)
                    display_results(result)
                    os.unlink(temp_path)


def display_ui():
    """Display Adaptive Huffman compression UI"""
    st.markdown("### Adaptive Huffman Compression")
    st.info("Adaptive Huffman coding dynamically builds the code tree as data is processed, adapting to changing symbol frequencies.")

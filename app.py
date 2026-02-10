import streamlit as st
import os
import tempfile
import shutil
from io import BytesIO
import pandas as pd
import base64

# Import compression modules
from compressor import compare_all_techniques_with_choice
from imageCompression import compare_all_image_techniques_with_choice
from audio_compression import AudioCompressor
from huffmanFunctions import _run_huffman, _run_huffman_image, huffmanImageCompression
from shanonfanofunctions import _run_shannon_fano, _run_shannon_fano_image, shannonImageCompression
from adaptiveHuffmanfunctions import _run_adaptive_huffman, _run_adaptive_huffman_image, adaptiveHuffmanImageCompression
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
        if algorithm == "huffman":
            result = _run_huffman(file_path)
        elif algorithm == "shannon":
            result = _run_shannon_fano(file_path)
        elif algorithm == "adaptive":
            result = _run_adaptive_huffman(file_path)
        else:
            return None
        
        return result
    except Exception as e:
        st.error(f"Compression error: {e}")
        return None

def compress_image_file(file_path, algorithm):
    """Compress image file using selected algorithm"""
    try:
        if algorithm == "huffman":
            result = _run_huffman_image(file_path)
        elif algorithm == "shannon":
            result = _run_shannon_fano_image(file_path)
        elif algorithm == "adaptive":
            result = _run_adaptive_huffman_image(file_path)
        else:
            return None
        
        return result
    except Exception as e:
        st.error(f"Image compression error: {e}")
        return None

def compress_audio_file(file_path, algorithm):
    """Compress audio file using selected algorithm"""
    try:
        compressor = AudioCompressor(base_output_dir=outputFiles)
        stats = compressor.compress_audio(file_path, algorithm=algorithm)
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
                                st.dataframe(df, use_container_width=True)
                                
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
                                st.dataframe(df, use_container_width=True)
                                
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
                                st.dataframe(df, use_container_width=True)
                                
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
                        # For decompression, we'd need to modify the decompressor functions
                        # to work with uploaded files instead of file selection
                        st.success(f"Decompressed using {algorithm.title()} algorithm!")
                        st.info("Download feature coming soon!")
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
                        st.success(f"Decompressed using {algorithm.title()} algorithm!")
                        st.info("Download feature coming soon!")
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
                        st.success(f"Decompressed using {algorithm.title()} algorithm!")
                        st.info("Download feature coming soon!")
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
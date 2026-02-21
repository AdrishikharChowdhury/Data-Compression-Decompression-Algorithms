import os

filePath = "./files"
inputFiles = f"{filePath}/inputs"
outputFiles = f"{filePath}/outputs"

# Huffman folders
outputHuffmanFiles = f"{outputFiles}/huffmann_files"
outputHuffmanText = f"{outputHuffmanFiles}/text"
outputHuffmanImage = f"{outputHuffmanFiles}/image"
outputHuffmanAudio = f"{outputHuffmanFiles}/audio"
outputHuffmanDecompressed = f"{outputHuffmanFiles}/decompressed"
outputHuffmanDecompressedText = f"{outputHuffmanDecompressed}/text"
outputHuffmanDecompressedImage = f"{outputHuffmanDecompressed}/image"
outputHuffmanDecompressedAudio = f"{outputHuffmanDecompressed}/audio"

# Shannon-Fano folders
outputShannonFiles = f"{outputFiles}/shannon_files"
outputShannonText = f"{outputShannonFiles}/text"
outputShannonImage = f"{outputShannonFiles}/image"
outputShannonAudio = f"{outputShannonFiles}/audio"
outputShannonDecompressed = f"{outputShannonFiles}/decompressed"
outputShannonDecompressedText = f"{outputShannonDecompressed}/text"
outputShannonDecompressedImage = f"{outputShannonDecompressed}/image"
outputShannonDecompressedAudio = f"{outputShannonDecompressed}/audio"

# Adaptive Huffman folders
outputAdaptiveHuffmannFiles = f"{outputFiles}/adaptive_huffman_files"
outputAdaptiveHuffmanText = f"{outputAdaptiveHuffmannFiles}/text"
outputAdaptiveHuffmanImage = f"{outputAdaptiveHuffmannFiles}/image"
outputAdaptiveHuffmanAudio = f"{outputAdaptiveHuffmannFiles}/audio"
outputAdaptiveHuffmanDecompressed = f"{outputAdaptiveHuffmannFiles}/decompressed"
outputAdaptiveHuffmanDecompressedText = f"{outputAdaptiveHuffmanDecompressed}/text"
outputAdaptiveHuffmanDecompressedImage = f"{outputAdaptiveHuffmanDecompressed}/image"
outputAdaptiveHuffmanDecompressedAudio = f"{outputAdaptiveHuffmanDecompressed}/audio"

# Create all directories
def create_all_directories():
    """Create all output directories with subfolder structure"""
    dirs = [
        outputHuffmanFiles, outputHuffmanText, outputHuffmanImage, outputHuffmanAudio,
        outputHuffmanDecompressed, outputHuffmanDecompressedText, outputHuffmanDecompressedImage, outputHuffmanDecompressedAudio,
        outputShannonFiles, outputShannonText, outputShannonImage, outputShannonAudio,
        outputShannonDecompressed, outputShannonDecompressedText, outputShannonDecompressedImage, outputShannonDecompressedAudio,
        outputAdaptiveHuffmannFiles, outputAdaptiveHuffmanText, outputAdaptiveHuffmanImage, outputAdaptiveHuffmanAudio,
        outputAdaptiveHuffmanDecompressed, outputAdaptiveHuffmanDecompressedText, outputAdaptiveHuffmanDecompressedImage, outputAdaptiveHuffmanDecompressedAudio
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

# Auto-create directories when module is imported
create_all_directories()

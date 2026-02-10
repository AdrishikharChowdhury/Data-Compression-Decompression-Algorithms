# Updated Folder Structure

## Overview
The output folder structure has been reorganized to have subfolders for text, image, and audio files, plus decompressed folders for each file type.

## New Structure

### Huffman Files (`./files/outputs/huffmann_files/`)
```
huffmann_files/
├── text/                    # Compressed text files (.huf)
├── image/                   # Compressed image files (.huf)
├── audio/                   # Compressed audio files (.huf)
└── decompressed/
    ├── text/                # Decompressed text files (.txt)
    ├── image/               # Decompressed image files (.jpg, .png, etc.)
    └── audio/               # Decompressed audio files (.wav)
```

### Shannon-Fano Files (`./files/outputs/shannon_files/`)
```
shannon_files/
├── text/                    # Compressed text files (.sf)
├── image/                   # Compressed image files (.sf)
├── audio/                   # Compressed audio files (.sf)
└── decompressed/
    ├── text/                # Decompressed text files (.txt)
    ├── image/               # Decompressed image files (.jpg, .png, etc.)
    └── audio/               # Decompressed audio files (.wav)
```

### Adaptive Huffman Files (`./files/outputs/adaptive_huffman_files/`)
```
adaptive_huffman_files/
├── text/                    # Compressed text files (.ahuf)
├── image/                   # Compressed image files (.ahuf)
├── audio/                   # Compressed audio files (.ahuf)
└── decompressed/
    ├── text/                # Decompressed text files (.txt)
    ├── image/               # Decompressed image files (.jpg, .png, etc.)
    └── audio/               # Decompressed audio files (.wav)
```

## Constants (from constants.py)

### Huffman Paths
- `outputHuffmanText` - `./files/outputs/huffmann_files/text`
- `outputHuffmanImage` - `./files/outputs/huffmann_files/image`
- `outputHuffmanAudio` - `./files/outputs/huffmann_files/audio`
- `outputHuffmanDecompressedText` - `./files/outputs/huffmann_files/decompressed/text`
- `outputHuffmanDecompressedImage` - `./files/outputs/huffmann_files/decompressed/image`
- `outputHuffmanDecompressedAudio` - `./files/outputs/huffmann_files/decompressed/audio`

### Shannon-Fano Paths
- `outputShannonText` - `./files/outputs/shannon_files/text`
- `outputShannonImage` - `./files/outputs/shannon_files/image`
- `outputShannonAudio` - `./files/outputs/shannon_files/audio`
- `outputShannonDecompressedText` - `./files/outputs/shannon_files/decompressed/text`
- `outputShannonDecompressedImage` - `./files/outputs/shannon_files/decompressed/image`
- `outputShannonDecompressedAudio` - `./files/outputs/shannon_files/decompressed/audio`

### Adaptive Huffman Paths
- `outputAdaptiveHuffmanText` - `./files/outputs/adaptive_huffman_files/text`
- `outputAdaptiveHuffmanImage` - `./files/outputs/adaptive_huffman_files/image`
- `outputAdaptiveHuffmanAudio` - `./files/outputs/adaptive_huffman_files/audio`
- `outputAdaptiveHuffmanDecompressedText` - `./files/outputs/adaptive_huffman_files/decompressed/text`
- `outputAdaptiveHuffmanDecompressedImage` - `./files/outputs/adaptive_huffman_files/decompressed/image`
- `outputAdaptiveHuffmanDecompressedAudio` - `./files/outputs/adaptive_huffman_files/decompressed/audio`

## Files Updated

1. **constants.py** - Added all new path constants and automatic directory creation
2. **main.py** - Updated to use new folder paths and imports
3. **compressor.py** - Updated to save text files to text subfolders
4. **huffmanFunctions.py** - Updated to save text and image files to appropriate subfolders
5. **shanonfanofunctions.py** - Updated to save text and image files to appropriate subfolders
6. **adaptiveHuffmanfunctions.py** - Updated to save text and image files to appropriate subfolders
7. **huffmanDecompressor.py** - Updated to read from text folder and save to decompressed/text
8. **shannonDecompressor.py** - Updated to read from text folder and save to decompressed/text
9. **adaptiveHuffmanDecompressor.py** - Updated to read from text folder and save to decompressed/text
10. **decompressor.py** - Updated selectCompressedFile() to support file_type parameter
11. **audio_compression.py** - Updated to save audio files to audio subfolders
12. **audioDecompressor.py** - Updated to read from audio folder and save to decompressed/audio

## Usage

Directories are automatically created when any module imports constants.py. Files are automatically saved to the correct subfolder based on their type:

- Text files → `text/` folder
- Image files → `image/` folder  
- Audio files → `audio/` folder
- Decompressed files → `decompressed/{type}/` folder

## Benefits

1. **Better Organization** - Files are organized by type within each algorithm folder
2. **Easier Management** - Easier to find and manage compressed/decompressed files
3. **Consistency** - Same structure across all three compression algorithms
4. **Separation** - Compressed and decompressed files are in separate folders

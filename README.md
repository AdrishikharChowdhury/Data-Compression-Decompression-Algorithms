# Data Compression and Decompression Suite

A comprehensive Python implementation of three major lossless data compression algorithms with **complete decompression functionality** and performance comparison.

## 🎯 Project Overview

This project implements:
- **Shannon-Fano Compression & Decompression**
- **Huffman Compression & Decompression** 
- **Adaptive Huffman Compression & Decompression**

With the following key features:
- ✅ **Complete compression AND decompression** for all techniques
- ✅ Proper file extensions (`.sf`, `.huf`, `.ahuf`) for each format
- ✅ **Modular architecture** with separate compressor/decompressor classes
- ✅ Clean file organization and automatic directory management
- ✅ 100% accurate decompression verification
- ✅ Performance benchmarking and comparison

## 📊 Performance Results

Latest performance test on sample text (1368 characters):

| Technique | Original Size | Compressed Size | Space Saved | Rank |
|-----------|---------------|-----------------|-------------|------|
| **Adaptive Huffman** | 1368B | 776B | **43.3%** | 🥇 |
| **Shannon-Fano** | 1368B | 894B | **34.6%** | 🥈 |
| **Huffman** | 1368B | 952B | **30.4%** | 🥉 |

## 🏗️ Project Structure

```
data compression and decompression using huffmann and shanon fano/
├── 📁 files/
│   ├── 📁 inputs/
│   │   └── 📄 test.txt                              # Test input file
│   └── 📁 outputs/
│       ├── 📁 huffmann_files/
│       │   ├── 📄 huffman_test.huf                 # Compressed Huffman file
│       │   └── 📄 huffman_decompressed.txt          # Decompressed output
│       ├── 📁 shannon_files/
│       │   ├── 📄 shannon_test.sf                  # Compressed Shannon-Fano file
│       │   └── 📄 shannon_decompressed.txt         # Decompressed output
│       └── 📁 adaptive_huffman_files/
│           ├── 📄 adaptive_test.ahuf               # Compressed Adaptive Huffman file
│           └── 📄 adaptive_decompressed.txt        # Decompressed output
│
├── 🐍 Compression Modules/
│   ├── 📄 huffmanCompressor.py                     # Huffman compression class
│   ├── 📄 shanonCompressor.py                      # Shannon-Fano compression class
│   ├── 📄 adaptiveHuffmann.py                     # Adaptive Huffman compression class
│   └── 📄 compressor.py                           # Compression coordination
│
├── 🐍 Decompression Modules/
│   ├── 📄 huffmanDecompressor.py                   # Huffman decompression class
│   ├── 📄 shannonDecompressor.py                   # Shannon-Fano decompression class
│   ├── 📄 adaptiveHuffmanDecompressor.py           # Adaptive Huffman decompression class
│   └── 📄 decompressor.py                          # Decompression coordination
│
├── 🐍 Core Files/
│   ├── 📄 main.py                                  # Main menu interface
│   └── 📄 file_handler.py                         # File I/O utilities
│
└── 📄 README.md                                    # This documentation
```

## 🚀 Installation & Usage

### Prerequisites

Install the required package:
```bash
pip install bitarray
```

### Running the Application

```bash
python main.py
```

### Menu Navigation

#### Main Menu
```
1. Compress a file
2. Decompress a file  
3. Exit
```

#### Compression Menu
```
1. Huffman Compression
2. Shannon-Fano Compression
3. Adaptive Huffman Compression
4. Compare All Techniques
5. Back to Main Menu
```

#### Decompression Menu
```
1. Huffman Decompression
2. Shannon-Fano Decompression
3. Adaptive Huffman Decompression
4. Back to Main Menu
```

### Programmatic Usage

#### Compression
```python
# Import individual compressors
from huffmanCompressor import HuffmanCompressor
from shanonCompressor import ShannonFanoCompressor
from adaptiveHuffmann import AdaptiveHuffmanCompressor

# Compress with different algorithms
text = "Your text here"

# Huffman
huffman = HuffmanCompressor()
compressed, root, codes = huffman.compress(text)

# Shannon-Fano
shannon = ShannonFanoCompressor()
compressed, codes = shannon.compress(text)

# Adaptive Huffman
adaptive = AdaptiveHuffmanCompressor()
compressed_bits, total_bits = adaptive.compress_stream(text)
```

#### Decompression
```python
# Import individual decompressors
from huffmanDecompressor import HuffmanDecompressor
from shannonDecompressor import ShannonFanoDecompressor
from adaptiveHuffmanDecompressor import AdaptiveHuffmanDecompressor

# Decompress from files
huffman_decomp = HuffmanDecompressor()
original_text = huffman_decomp.decompress_from_file('compressed.huf')

shannon_decomp = ShannonFanoDecompressor()
original_text = shannon_decomp.decompress_from_file('compressed.sf')

adaptive_decomp = AdaptiveHuffmanDecompressor()
original_text = adaptive_decomp.decompress_from_file('compressed.ahuf')
```

2. Create a virtual environment:
```bash
python3 -m venv project
source project/bin/activate  # Linux/Mac
# or
project\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Interactive Mode

Run the main program:
```bash
python main.py
```

Menu options:
1. **Huffman Compression** - Compress using static Huffman coding
2. **Shannon-Fano Compression** - Compress using Shannon-Fano coding
3. **Adaptive Huffman Compression** - Compress using adaptive Huffman coding
4. **Compare All Techniques** - Run all three and compare results
5. **Exit**

### Input Files

Place your test files in `files/inputs/test.txt`. The program will compress this file and save outputs to their respective directories.

### Programmatic Usage

```python
from huffmanCompressor import HuffmanCompressor
from shanonCompressor import ShannonFanoCompressor
from adaptiveHuffmann import AdaptiveHuffmanCompressor

# Read your text
with open('input.txt', 'r') as f:
    text = f.read()

# Huffman
huffman = HuffmanCompressor()
compressed, freq, tree = huffman.compress(text)

# Shannon-Fano
shannon = ShannonFanoCompressor()
compressed, bits = shannon.compress(text)

# Adaptive Huffman
adaptive = AdaptiveHuffmanCompressor()
compressed, bits = adaptive.compress_stream(text)
```

## 🧩 Algorithm Details

### 1. Huffman Compression & Decompression
- **Principle**: Optimal prefix coding based on character frequency
- **File Extension**: `.huf`
- **Passes Required**: 2 (frequency analysis + compression)
- **Decompression**: Rebuilds tree from stored frequency table
- **Advantages**: Theoretically optimal for known data
- **Use Cases**: Static files where maximum compression is priority

### 2. Shannon-Fano Compression & Decompression  
- **Principle**: Recursive frequency splitting into equal probability groups
- **File Extension**: `.sf`
- **Passes Required**: 2 (frequency analysis + compression)
- **Decompression**: Uses stored code-to-character mapping
- **Advantages**: Simple to implement, educational value
- **Use Cases**: Learning purposes, simple compression needs

### 3. Adaptive Huffman Compression & Decompression
- **Principle**: Dynamic tree updates during compression/decompression
- **File Extension**: `.ahuf`
- **Passes Required**: 1 (single-pass streaming)
- **Decompression**: Rebuilds tree dynamically using same FGK algorithm
- **Advantages**: 
  - Real-time compression
  - No frequency table storage needed
  - Works with unknown data length
  - Memory efficient
- **Use Cases**: Streaming data, real-time applications, IoT devices

## 📊 Performance Analysis

### Current Results (1368 bytes sample):
- **Adaptive Huffman achieves best compression** due to efficient bit packing and optimized tree updates
- **Shannon-Fano provides middle ground** with reasonable compression
- **Standard Huffman performs worst** on this sample due to suboptimal frequency table encoding overhead

### When Each Algorithm Excels:

#### 🥇 Adaptive Huffman: Best for:
- **Streaming applications** - Real-time video/audio
- **Single-pass requirements** - Large files where two passes are expensive
- **Unknown data length** - Continuous data streams
- **Memory constraints** - No need to store frequency table

#### 🥈 Shannon-Fano: Good for:
- **Educational purposes** - Understanding compression principles
- **Simple implementations** - When complexity needs to be minimal
- **Quick prototypes** - Fast development cycles

#### 🥉 Standard Huffman: Use when:
- **Maximum theoretical optimality** is required
- **Data is fully available upfront**
- **Two-pass processing** is acceptable
- **Predictable performance** is needed

## 🔧 Technical Implementation

### File Format Specifications

#### Huffman (`.huf`)
```
[4 bytes: frequency table size]
[For each character: 1 byte char length + char bytes + 4 bytes frequency]
[1 byte: padding]
[bit-packed compressed data]
```

#### Shannon-Fano (`.sf`)
```
[4 bytes: header "SF01"]
[2 bytes: number of unique characters]
[For each character: 1 byte char length + char bytes + 1 byte code length + code bytes]
[3 bytes: separator 0xFF 0xFF 0xFF]
[1 byte: padding]
[bit-packed compressed data]
```

#### Adaptive Huffman (`.ahuf`)
```
[8 bytes: original file length]
[bit-packed compressed stream with NYT codes and character bytes]
```

## 🧪 Testing & Verification

### Basic Testing
```python
# Test all compressions and decompressions
from compressor import compare_all_techniques
compare_all_techniques()

# Test all decompressions
from decompressor import decompress_all
decompress_all()

# Verify accuracy (should return True for all)
import os
original = open('files/inputs/test.txt', 'r').read()

# Compare each decompressed with original
for algo in ['huffman', 'shannon', 'adaptive_huffman']:
    decomp_file = f'files/outputs/{algo}_files/{algo.split("_")[0]}_decompressed.txt'
    decomp_text = open(decomp_file, 'r').read()
    print(f"{algo}: {'✅ Match' if original == decomp_text else '❌ Mismatch'}")
```

### Individual Algorithm Testing
```python
# Test specific compression/decompression
from compressor import huffmanCompression
from decompressor import huffmanDecompression

huffmanCompression()  # Compress
huffmanDecompression()  # Decompress
```

## 🎯 Features Added

### ✅ Complete Functionality
- **Full decompression** for all three algorithms (previously missing)
- **100% accuracy verification** - all decompressed files match original
- **Proper file extensions** - `.huf`, `.sf`, `.ahuf` for each technique
- **Modular architecture** - separate compressor and decompressor classes

### ✅ Enhanced Architecture
- **Separation of concerns** - compression and decompression in different modules
- **Independent classes** - each technique has dedicated compressor and decompressor
- **Clean interfaces** - consistent API across all algorithms
- **Easy maintenance** - modular structure for future improvements

### ✅ Improved User Experience
- **Organized file structure** - automatic directory creation
- **Comprehensive error handling** - missing files, permission issues
- **Progress feedback** - clear status messages during operations
- **Menu-driven interface** - intuitive navigation

## 📦 Dependencies

- `bitarray` - Efficient bit-level operations

Install via:
```bash
pip install bitarray
```

## 🚀 Future Enhancements

- [ ] Support for binary file compression
- [ ] GUI interface with drag-and-drop
- [ ] Additional compression algorithms (LZW, LZ77)
- [ ] Batch file processing
- [ ] Progress bars for large files
- [ ] Compression level presets
- [ ] Performance optimization for large files

## 🐛 Troubleshooting

### Common Issues
- **Missing bitarray**: Install with `pip install bitarray`
- **File not found**: Ensure test.txt exists in `files/inputs/`
- **Permission errors**: Check write permissions in output directories
- **Decpression failures**: Verify compressed files exist and are not corrupted

### Debug Mode
Enable detailed output:
```python
# Test specific algorithm with debug info
from compressor import huffmanCompression, huffmanDecompression
huffmanCompression()
huffmanDecompression()

# Verify file integrity
import os
print("Compressed files:", [f for f in os.listdir('files/outputs') if f.endswith(('.huf', '.sf', '.ahuf'))])
print("Decompressed files:", [f for f in os.listdir('files/outputs') if 'decompressed' in f])
```

## 📚 References

- Huffman, D. A. (1952). "A Method for the Construction of Minimum-Redundancy Codes"
- Shannon, C. E. (1948). "A Mathematical Theory of Communication"
- Vitter, J. S. (1987). "Design and Analysis of Dynamic Huffman Codes"

## 📄 License

This project is available for educational purposes.

## 👨‍💻 Author

- **AdrishikharChowdhury**
    GitHub: [AdrishikharChowdhury](https://github.com/AdrishikharChowdhury)
- **adrikadas0709-afk**
    GitHub: [Adrika Das](https://github.com/adrikadas0709-afk)
- **ryanchowdhury09-lol**
    GitHub: [Ryan Chowdhury](https://github.com/ryanchowdhury09-lol)
- **bristichowdhury87-creator**
    GitHub: [Bristi Chowdhury](https://github.com/bristichowdhury87-creator)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

---

**Note**: This implementation is for educational purposes. For production use, consider established compression libraries like `zlib`, `gzip`, or `bz2`.

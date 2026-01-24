# Data Compression Algorithms

A Python implementation of three popular lossless compression algorithms: **Huffman Coding**, **Shannon-Fano Coding**, and **Adaptive Huffman Coding**.

## 📋 Overview

This project demonstrates and compares three data compression techniques:

- **Huffman Compression** - Optimal prefix-free coding using a binary tree
- **Shannon-Fano Compression** - Top-down approach to prefix coding
- **Adaptive Huffman Compression** - Dynamic tree-building for streaming data (FGK algorithm)

## 🚀 Features

- ✅ Three compression algorithms implemented from scratch
- ✅ Side-by-side comparison of compression ratios
- ✅ Support for text and binary files
- ✅ Interactive menu-driven interface
- ✅ Detailed compression statistics

## 📁 Project Structure

```
.
├── adaptiveHuffmann.py      # Adaptive Huffman (FGK) implementation
├── huffmanCompressor.py     # Static Huffman compression
├── shanonCompressor.py      # Shannon-Fano compression
├── compressor.py            # Main compression logic and comparison
├── file_handler.py          # File I/O utilities
├── main.py                  # Interactive menu interface
├── requirements.txt         # Python dependencies
├── files/
│   ├── inputs/             # Input test files
│   └── outputs/            # Compressed output files
│       ├── huffmann_files/
│       ├── shannon_files/
│       └── adaptive_huffman_files/
└── project/                # Python virtual environment
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd "data compression and decompression using huffmann and shanon fano"
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

## 📊 Compression Results

Example comparison on text data (2,300 bytes):

| Technique | Original Size | Compressed Size | Space Saved | Speed |
|-----------|--------------|-----------------|-------------|-------|
| **Huffman** | 2300B | 1372B | **40.3%** | ⚡ Fast |
| **Shannon-Fano** | 2300B | 1386B | 39.7% | ⚡ Fast |
| **Adaptive Huffman** | 2300B | 1445B | 37.2% | 🐌 Slower |

### When to Use Each Algorithm

#### Static Huffman
- ✅ Best compression ratio for static files
- ✅ Fast compression and decompression
- ❌ Requires two passes (analyze + compress)
- **Use for**: Files you can read entirely, maximum compression

#### Shannon-Fano
- ✅ Simple to implement
- ✅ Fast compression
- ❌ Slightly worse than Huffman
- **Use for**: Educational purposes, simple compression needs

#### Adaptive Huffman
- ✅ Single-pass compression (streaming)
- ✅ No need to transmit frequency table
- ✅ Adapts to changing character distributions
- ❌ ~2-5% worse compression ratio
- ❌ Slower due to tree updates
- **Use for**: Live streaming, real-time data, unknown data length

## 🔬 Algorithm Details

### Huffman Coding
- Two-pass algorithm: frequency analysis + encoding
- Builds optimal prefix-free binary tree
- Guarantees minimum average code length
- Time complexity: O(n log n)

### Shannon-Fano Coding
- Top-down recursive partitioning
- Divides symbols by cumulative probability
- Near-optimal but not guaranteed optimal
- Time complexity: O(n log n)

### Adaptive Huffman (FGK)
- Single-pass, online algorithm
- Dynamically updates tree using Vitter's algorithm
- Maintains sibling property for optimal tree
- Ideal for streaming data
- Time complexity: O(n log n) with higher constant

## 🧪 Testing

Test with included samples:
```bash
# Test on text file
python -c "from compressor import compare_all_techniques; compare_all_techniques()"

# Test on audio data (synthetic)
python -c "
from adaptiveHuffmann import AdaptiveHuffmanCompressor
with open('files/inputs/synthetic_audio.bin', 'rb') as f:
    data = f.read().decode('latin-1')
adaptive = AdaptiveHuffmanCompressor()
compressed, bits = adaptive.compress_stream(data)
print(f'Compressed {len(data)} bytes to {bits//8} bytes')
"
```

## 📦 Dependencies

- `bitarray` - Efficient bit-level operations

Install via:
```bash
pip install bitarray
```

## 🎯 Future Enhancements

- [ ] Decompression implementations
- [ ] Support for larger files (chunked processing)
- [ ] GUI interface
- [ ] Performance optimizations
- [ ] LZW and other compression algorithms
- [ ] Benchmark suite with various file types

## 🐛 Known Issues

- Adaptive Huffman is significantly slower on large files due to tree rebalancing
- Binary files may have poor compression ratios (expected for high-entropy data)

## 📚 References

- Huffman, D. A. (1952). "A Method for the Construction of Minimum-Redundancy Codes"
- Shannon, C. E. (1948). "A Mathematical Theory of Communication"
- Vitter, J. S. (1987). "Design and Analysis of Dynamic Huffman Codes"

## 📄 License

This project is available for educational purposes.

## 👨‍💻 Author

**AdrishikharChowdhury**
- GitHub: [AdrishikharChowdhury](https://github.com/AdrishikharChowdhury)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

---

**Note**: This implementation is for educational purposes. For production use, consider established compression libraries like `zlib`, `gzip`, or `bz2`.

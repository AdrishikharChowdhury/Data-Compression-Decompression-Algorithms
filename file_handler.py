from bitarray import bitarray
# file_handler.py - Read/write utilities
def read_text_file(filename):
    """Read text file → string"""
    with open(filename, 'r') as f:
        return f.read()

def write_text_file(filename, content):
    """Write string → text file"""
    with open(filename, 'w') as f:
        f.write(content)

def read_binary_file(filename):
    """Read ANY file → bytes"""
    with open(filename, 'rb') as f:
        return f.read()

def save_compressed_file(compressed_bits, input_filename, output_filename):
    orig_len = len(open(input_filename, 'rb').read())
    bits = bitarray(compressed_bits)
    
    with open(output_filename, 'wb') as f:
        f.write(orig_len.to_bytes(8, 'big'))
        bits.tofile(f)

def save_compressed_huffman_file(compressed_bits, input_filename, output_filename):
    from huffmanCompressor import HuffmanCompressor
    text = read_text_file(input_filename)
    compressor = HuffmanCompressor()
    compressor.compress_file(text, output_filename)

def save_compressed_shannon_file(compressed_bits, input_filename, output_filename):
    from shanonCompressor import ShannonFanoCompressor
    text = read_text_file(input_filename)
    compressor = ShannonFanoCompressor()
    compressor.compress_file(text, output_filename)

def read_binary_data(filename):
    """Read ANY file as bytes and return as list of byte values"""
    with open(filename, 'rb') as f:
        return list(f.read())

def save_compressed_binary_file(compressed_bits, input_filename, output_filename, freq_dict=None, codes_dict=None, method='huffman'):
    """Save compressed binary data with appropriate header"""
    from bitarray import bitarray
    orig_len = len(open(input_filename, 'rb').read())
    bits = bitarray(compressed_bits)
    
    with open(output_filename, 'wb') as f:
        f.write(orig_len.to_bytes(8, 'big'))
        
        if method == 'huffman' and freq_dict:
            f.write(len(freq_dict).to_bytes(4, 'big'))
            for char, f_val in freq_dict.items():
                if isinstance(char, str):
                    char_byte = ord(char)
                else:
                    char_byte = char
                f.write(char_byte.to_bytes(1, 'big'))
                capped_freq = min(f_val, 2**32 - 1)
                f.write(capped_freq.to_bytes(4, 'big'))
        elif method == 'shannon' and codes_dict:
            f.write(len(codes_dict).to_bytes(2, 'big'))
            for byte_val, code in codes_dict.items():
                # byte_val should be a character (chr(byte))
                if isinstance(byte_val, str):
                    char_byte = ord(byte_val)
                else:
                    char_byte = byte_val
                f.write(char_byte.to_bytes(1, 'big'))
                f.write(len(code).to_bytes(1, 'big'))
                code_int = int(code, 2)
                code_bytes = (len(code) + 7) // 8
                capped_code = min(code_int, 2**(8*code_bytes) - 1)
                f.write(capped_code.to_bytes(code_bytes, 'big'))
        
        padding = (8 - len(bits) % 8) % 8
        if padding > 0:
            bits.extend([0] * padding)
        f.write(padding.to_bytes(1, 'big'))
        bits.tofile(f)


def _print_results(stats):
    """Helper function to print formatted compression results."""
    if not stats:
        return
    savings = (1 - stats["comp_size"] / stats["orig_size"]) * 100
    print(
        f"Compression complete for {stats['name']}. "
        f"Original: {stats['orig_size']}B, "
        f"Compressed: {stats['comp_size']}B, "
        f"Space Saved: {savings:.1f}%"
    )

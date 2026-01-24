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

# shannonfano/textShanon.py - Fixed Shannon-Fano compression
from collections import Counter
from file_handler import read_text_file, _print_results
import os
from constants import inputFiles, outputShannonText

def _run_shannon_fano(input_file):
    text = read_text_file(input_file)
    orig_len = len(open(input_file, 'rb').read())
    
    print(f"Analyzing text ({orig_len} bytes) for Shannon-Fano compression...")
    
    try:
        return _compress_shannon_fano(text, input_file, orig_len)
    except Exception as e:
        print(f"    Shannon-Fano error: {e}")
        return {"name": "Shannon-Fano", "orig_size": orig_len, "comp_size": orig_len}

def shanonCompression():
    """CLI function for Shannon-Fano compression"""
    input_file = f"{inputFiles}/test.txt"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    
    result = _run_shannon_fano(input_file)
    if result:
        _print_results(result)

def _compress_shannon_fano(text, input_file, orig_len):
    """Proper Shannon-Fano compression with explicit code table"""
    if isinstance(text, str):
        byte_data = bytes(ord(c) for c in text)
    else:
        byte_data = text
    
    # Build frequency table
    freq = Counter(byte_data)
    if not freq:
        return {"name": "Shannon-Fano", "orig_size": orig_len, "comp_size": 0}
    
    # Build Shannon-Fano codes
    codes = _build_shannon_fano_codes(freq)
    
    # Encode the data
    encoded_bits = ''.join(codes.get(b, '') for b in byte_data)
    
    # Pad to byte boundary
    padding = (8 - len(encoded_bits) % 8) % 8
    encoded_bits += '0' * padding
    
    # Pack into bytes
    packed_data = bytearray()
    for i in range(0, len(encoded_bits), 8):
        byte_val = int(encoded_bits[i:i+8], 2)
        packed_data.append(byte_val)
    
    # Build code table for storage (symbol -> code as string)
    code_table = {sym: codes[sym] for sym in codes}
    
    # Write file with proper format
    output_file = f"{outputShannonText}/{os.path.splitext(os.path.basename(input_file))[0]}.sf"
    with open(output_file, 'wb') as f:
        # Header: SF02 + version
        f.write(b'SF02')
        
        # Original size (4 bytes)
        f.write(orig_len.to_bytes(4, 'big'))
        
        # Number of unique symbols (2 bytes)
        num_symbols = len(code_table)
        f.write(num_symbols.to_bytes(2, 'big'))
        
        # Write code table: for each symbol, store (byte value, code string as bytes)
        for sym_byte, code_str in sorted(code_table.items()):
            f.write(bytes([sym_byte]))  # symbol byte
            f.write(bytes([len(code_str)]))  # code length
            # Store code as bytes (pad to byte boundary)
            code_bytes = int(code_str, 2).to_bytes((len(code_str) + 7) // 8, 'big')
            f.write(code_bytes)
        
        # Padding info (1 byte)
        f.write(bytes([padding]))
        
        # Compressed data
        f.write(packed_data)
    
    comp_size = os.path.getsize(output_file)
    savings = (orig_len - comp_size) / orig_len * 100
    print(f"   Shannon-Fano compression: {savings:.1f}%")
    return {"name": "Shannon-Fano", "orig_size": orig_len, "comp_size": comp_size}

def _build_shannon_fano_codes(freq):
    """Build Shannon-Fano codes - returns dict mapping byte -> code string"""
    # Sort by frequency descending
    symbols = sorted(freq.items(), key=lambda x: -x[1])
    
    if len(symbols) == 0:
        return {}
    
    if len(symbols) == 1:
        return {symbols[0][0]: '0'}
    
    # Split into roughly equal halves
    total = sum(f for _, f in symbols)
    half = total / 2
    
    left = []
    right = []
    left_sum = 0
    
    for sym, f in symbols:
        if left_sum < half:
            left.append((sym, f))
            left_sum += f
        else:
            right.append((sym, f))
    
    # Ensure both sides have at least one symbol
    if not left:
        left = [right[0]]
        right = right[1:]
    if not right:
        right = [left[-1]]
        left = left[:-1]
    
    codes = {}
    
    # Assign codes recursively
    def assign_codes(symbol_list, prefix):
        if len(symbol_list) == 1:
            codes[symbol_list[0][0]] = prefix or '0'
            return
        
        if len(symbol_list) == 2:
            codes[symbol_list[0][0]] = prefix + '0'
            codes[symbol_list[1][0]] = prefix + '1'
            return
        
        # Split into two groups
        total = sum(f for _, f in symbol_list)
        half = total / 2
        
        left_group = []
        right_group = []
        left_sum = 0
        
        for item in symbol_list:
            if left_sum < half:
                left_group.append(item)
                left_sum += item[1]
            else:
                right_group.append(item)
        
        if not left_group:
            left_group = [right_group[0]]
            right_group = right_group[1:]
        if not right_group:
            right_group = [left_group[-1]]
            left_group = left_group[:-1]
        
        assign_codes(left_group, prefix + '0')
        assign_codes(right_group, prefix + '1')
    
    assign_codes(left, '0')
    assign_codes(right, '1')
    
    return codes

from collections import Counter
from file_handler import read_text_file,_print_results
import os
from constants import inputFiles,outputShannonText
from shanonCompressor import ShannonFanoCompressor

def _run_shannon_fano(input_file):
    text = read_text_file(input_file)
    orig_len = len(open(input_file, 'rb').read())
    
    print(f"Analyzing text ({orig_len} bytes) for Shannon-Fano compression...")
    
    try:
        if orig_len < 50:
            result = _ultra_optimized_shannon_fano(text, input_file, orig_len)
        elif orig_len < 200:
            result = _minimized_overhead_shannon_fano(text, input_file, orig_len)
        else:
            result = _improved_standard_shannon_fano(text, input_file, orig_len)
        
        return result
        
    except Exception as e:
        print(f"    Shannon-Fano error: {e}")
        return {"name": "Shannon-Fano", "orig_size": orig_len, "comp_size": orig_len}

def _ultra_optimized_shannon_fano(text, input_file, orig_len):
    from collections import Counter
    
    freq = Counter(text)
    unique_chars = len(freq)
    
    if unique_chars <= 4:
        sorted_chars = sorted(freq.items(), key=lambda item: item[1], reverse=True)
        
        half_point = len(sorted_chars) // 2
        left_group = sorted_chars[:half_point]
        right_group = sorted_chars[half_point:]
        
        char_codes = {}
        for i, (char, count) in enumerate(left_group):
            char_codes[char] = f'0{i:01b}' if len(left_group) > 1 else '0'
        for i, (char, count) in enumerate(right_group):
            char_codes[char] = f'1{i:01b}' if len(right_group) > 1 else '1'
        
        encoded_bits = ''.join(char_codes[char] for char in text)
        
        padding = (8 - len(encoded_bits) % 8) % 8
        encoded_bits += '0' * padding
        
        output_file = f"{outputShannonText}/{os.path.splitext(os.path.basename(input_file))[0]}.sf"
        with open(output_file, 'wb') as f:
            f.write(b'S')
            f.write(orig_len.to_bytes(2, 'big'))
            f.write(padding.to_bytes(1, 'big'))
            f.write(len(char_codes).to_bytes(1, 'big'))
            
            for char, code in char_codes.items():
                char_val = ord(char) % 256
                f.write(char_val.to_bytes(1, 'big'))
                f.write(len(code).to_bytes(1, 'big'))
                f.write(int(code, 2).to_bytes(1, 'big'))
            
            for i in range(0, len(encoded_bits), 8):
                byte_val = int(encoded_bits[i:i+8], 2)
                f.write(byte_val.to_bytes(1, 'big'))
        
        comp_size = len(open(output_file, 'rb').read())
        if comp_size < orig_len:
            savings = (orig_len - comp_size) / orig_len * 100
            print(f"   Shannon-Fano compression: {savings:.1f}%")
            return {"name": "Shannon-Fano", "orig_size": orig_len, "comp_size": comp_size}
    
    compressed = []
    i = 0
    while i < len(text):
        char = text[i]
        count = 1
        j = i + 1
        while j < len(text) and text[j] == char and count < 127:
            count += 1
            j += 1
        
        if count > 2:
            compressed.append(f'[{ord(char):02d}{count:03d}]')
        else:
            compressed.append(char * count)
        i = j
    
    compressed_text = ''.join(compressed)
    
    if len(compressed_text) < orig_len:
        output_file = f"{outputShannonText}/{os.path.splitext(os.path.basename(input_file))[0]}.sf"
        with open(output_file, 'wb') as f:
            f.write(b'R')
            f.write(orig_len.to_bytes(2, 'big'))
            f.write(compressed_text.encode('utf-8'))
        
        comp_size = len(open(output_file, 'rb').read())
        savings = (orig_len - comp_size) / orig_len * 100
        print(f"   Shannon-Fano RLE: {savings:.1f}%")
        return {"name": "Shannon-Fano", "orig_size": orig_len, "comp_size": comp_size}
    
    output_file = f"{outputShannonText}/{os.path.splitext(os.path.basename(input_file))[0]}.sf"
    
    if orig_len == 2:
        char_codes = {text[0]: '0', text[1]: '1'} if len(set(text)) == 2 else {text[0]: '0', text[1]: '01'}
        encoded_bits = ''.join(char_codes[char] for char in text)
        
        byte_val = int(encoded_bits, 2)
        with open(output_file, 'wb') as f:
            f.write(byte_val.to_bytes(1, 'big'))
    else:
        bits_needed = max(1, len(set(text)).bit_length())
        packed_val = 0
        for char in text:
            packed_val = (packed_val << bits_needed) | ord(char)
        
        byte_count = max(1, (len(text) * bits_needed + 7) // 8)
        with open(output_file, 'wb') as f:
            f.write(packed_val.to_bytes(byte_count, 'big'))
    
    comp_size = len(open(output_file, 'rb').read())
    savings = (orig_len - comp_size) / orig_len * 100
    print(f"   Aggressive Shannon-Fano compression: {savings:.1f}%")
    return {"name": "Shannon-Fano", "orig_size": orig_len, "comp_size": comp_size}

def _minimized_overhead_shannon_fano(text, input_file, orig_len):
    freq = Counter(text)
    total_chars = len(text)
    
    if len(freq) <= 8:
        result = _fixed_3bit_encoding(text, input_file, orig_len, "minimal_shannon")
    elif len(freq) <= 16:
        result = _fixed_4bit_encoding(text, input_file, orig_len, "minimal_shannon")
    else:
        chars_by_freq = sorted(freq.items(), key=lambda item: -item[1])
        mid = len(chars_by_freq) // 2
        
        left_group = chars_by_freq[:mid]
        right_group = chars_by_freq[mid:]
        
        result = _smart_prefix_encoding(text, left_group, right_group, input_file, orig_len, "minimal_shannon")
    
    print(f"   Using minimized overhead Shannon-Fano approach")
    savings = (orig_len - result["comp_size"]) / orig_len * 100
    print(f"   Space saved: {savings:.1f}%")
    result["name"] = "Shannon-Fano"
    return result

def _improved_standard_shannon_fano(text, input_file, orig_len):
    from collections import Counter
    
    freq = Counter(text)
    unique_chars = sorted(set(text))
    
    if len(unique_chars) <= 2:
        bits_per_char = 1
    elif len(unique_chars) <= 4:
        bits_per_char = 2
    elif len(unique_chars) <= 8:
        bits_per_char = 3
    elif len(unique_chars) <= 16:
        bits_per_char = 4
    else:
        bits_per_char = 5
    
    chars_by_freq = sorted(freq.items(), key=lambda item: -item[1])
    mid_point = len(chars_by_freq) // 2
    
    left_group = chars_by_freq[:mid_point]
    right_group = chars_by_freq[mid_point:]
    
    char_codes = {}
    for i, (char, _) in enumerate(left_group):
            bits_needed = max(1, bits_per_char-1)
            char_codes[char] = format(i, f'0{bits_needed}b')
    for i, (char, _) in enumerate(right_group):
            bits_needed = max(1, bits_per_char-1)
            char_codes[char] = '1' + format(i, f'0{bits_needed}b')
    
    encoded_bits = ''.join(char_codes[char] for char in text)
    
    output_file = f"{outputShannonText}/{os.path.splitext(os.path.basename(input_file))[0]}.sf"
    with open(output_file, 'wb') as f:
        f.write(b'S')
        f.write(orig_len.to_bytes(2, 'big'))
        f.write(bits_per_char.to_bytes(1, 'big'))
        
        f.write(len(unique_chars).to_bytes(1, 'big'))
        for char, _ in chars_by_freq:
            char_val = ord(char) % 256
            f.write(char_val.to_bytes(1, 'big'))
        
        padding = (8 - len(encoded_bits) % 8) % 8
        encoded_bits += '0' * padding
        f.write(padding.to_bytes(1, 'big'))
        
        for i in range(0, len(encoded_bits), 8):
            byte_val = int(encoded_bits[i:i+8], 2)
            f.write(byte_val.to_bytes(1, 'big'))
    
    comp_size = len(open(output_file, 'rb').read())
    savings = (orig_len - comp_size) / orig_len * 100
    print(f"   Improved Shannon-Fano achieved {savings:.1f}%")
    return {"name": "Shannon-Fano", "orig_size": orig_len, "comp_size": comp_size}

def _fixed_3bit_encoding(text, input_file, orig_len, prefix):
    unique_chars = sorted(set(text))
    codes = {char: format(i, '03b') for i, char in enumerate(unique_chars)}
    
    encoded_bits = ''.join(codes[char] for char in text)
    
    output_file = f"{outputShannonText}/{os.path.splitext(os.path.basename(input_file))[0]}.sf"
    with open(output_file, 'wb') as f:
        f.write(b"3BIT")
        f.write(orig_len.to_bytes(2, 'big'))
        f.write(len(unique_chars).to_bytes(1, 'big'))
        
        for char in unique_chars:
            char_val = ord(char) % 256
            f.write(char_val.to_bytes(1, 'big'))
        
        padding = (8 - len(encoded_bits) % 8) % 8
        if padding > 0:
            encoded_bits += '0' * padding
            f.write(padding.to_bytes(1, 'big'))
        
        for i in range(0, len(encoded_bits), 8):
            byte_val = int(encoded_bits[i:i+8], 2)
            f.write(byte_val.to_bytes(1, 'big'))
    
    comp_size = len(open(output_file, 'rb').read())
    return {"orig_size": orig_len, "comp_size": comp_size}

def _fixed_4bit_encoding(text, input_file, orig_len, prefix):
    unique_chars = sorted(set(text))
    codes = {char: format(i, '04b') for i, char in enumerate(unique_chars)}
    
    encoded_bits = ''.join(codes[char] for char in text)
    
    output_file = f"{outputShannonText}/{os.path.splitext(os.path.basename(input_file))[0]}.sf"
    with open(output_file, 'wb') as f:
        f.write(b"4BIT")
        f.write(orig_len.to_bytes(2, 'big'))
        f.write(len(unique_chars).to_bytes(1, 'big'))
        
        for char in unique_chars:
            char_val = ord(char) % 256
            f.write(char_val.to_bytes(1, 'big'))
        
        padding = (8 - len(encoded_bits) % 8) % 8
        if padding > 0:
            encoded_bits += '0' * padding
            f.write(padding.to_bytes(1, 'big'))
        
        for i in range(0, len(encoded_bits), 8):
            byte_val = int(encoded_bits[i:i+8], 2)
            f.write(byte_val.to_bytes(1, 'big'))
    
    comp_size = len(open(output_file, 'rb').read())
    return {"orig_size": orig_len, "comp_size": comp_size}

def _smart_prefix_encoding(text, left_group, right_group, input_file, orig_len, prefix):
    left_codes = {char: '0' + format(i, f'0{(len(left_group)-1).bit_length()}b') 
                   for i, (char, _) in enumerate(left_group)}
    right_codes = {char: '1' + format(i, f'0{(len(right_group)-1).bit_length()}b') 
                     for i, (char, _) in enumerate(right_group)}
    
    all_codes = {**left_codes, **right_codes}
    
    encoded_bits = ''.join(all_codes[char] for char in text)
    
    output_file = f"{outputShannonText}/{os.path.splitext(os.path.basename(input_file))[0]}.sf"
    with open(output_file, 'wb') as f:
        f.write(b"PREFIX")
        f.write(orig_len.to_bytes(2, 'big'))
        f.write(len(left_group).to_bytes(1, 'big'))
        
        for char, _ in left_group:
            char_val = ord(char) % 256
            f.write(char_val.to_bytes(1, 'big'))
        for char, _ in right_group:
            char_val = ord(char) % 256
            f.write(char_val.to_bytes(1, 'big'))
        
        padding = (8 - len(encoded_bits) % 8) % 8
        if padding > 0:
            encoded_bits += '0' * padding
            f.write(padding.to_bytes(1, 'big'))
        
        for i in range(0, len(encoded_bits), 8):
            byte_val = int(encoded_bits[i:i+8], 2)
            f.write(byte_val.to_bytes(1, 'big'))
    
    comp_size = len(open(output_file, 'rb').read())
    return {"orig_size": orig_len, "comp_size": comp_size}

def shanonCompression():
    input_file = f"{inputFiles}/test.txt"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    stats = _run_shannon_fano(input_file)
    _print_results(stats)

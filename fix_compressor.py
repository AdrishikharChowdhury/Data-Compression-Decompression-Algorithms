#!/usr/bin/env python3

# Fix the duplicate print line issue in compressor.py

import fileinput
import sys

# Read the current file
with open('/home/shibaji/Downloads/My Workspace/Data-Compression-Decompression-Algorithms/compressor.py', 'r') as f:
    lines = f.readlines()

# Find and fix the duplicate print line
fixed_lines = []
for i, line in enumerate(lines, 1):
    if i == 108 and 'print("=" * 80)' in line:
        # This is the duplicate header line - remove it
        continue
    fixed_lines.append(line)

# Write back the fixed content
with open('/home/shibaji/Downloads/My Workspace/Data-Compression-Decompression-Algorithms/compressor.py', 'w') as f:
    f.writelines(fixed_lines)

print("Fixed the duplicate print line in compressor.py")
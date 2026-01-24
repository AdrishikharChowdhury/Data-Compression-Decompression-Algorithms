from compressor import huffmanCompression, shanonCompression, adaptiveHuffmanCompression, compare_all_techniques

def compressionChoice():
    print("Choose your compression choice:")
    print("1. Huffman\n2. Shannon\n3. Adaptive Huffman\n4. Comparison of All Techniques\n5. Exit\n")
    choice = int(input("Your choice: "))
    
    match choice:
        case 1:
            huffmanCompression()
        case 2:
            shanonCompression()
        case 3:
            adaptiveHuffmanCompression()
        case 4:
            compare_all_techniques()
        case 5:
            print("Thank you for using this program")
            return False
        case _:
            print("Invalid choice. Please try again.")
    return True


if __name__ == "__main__":
    while compressionChoice():  # Loop until exit
        pass

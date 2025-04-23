import numpy as np

def vector_preprocess(vector, dtype, name):
    """
    This function preprocesses a vector for CIM (Compute In Memory) operations.
    It splits a vector longer than 32 elements into several 32-dimensional sub-vectors,
    converts each sub-vector to an unsigned 8-bit (for int8) or 32-bit (for int32) representation,
    and then processes each sub-vector bitwise. For each sub-vector, it extracts bits from each element,
    inverts them, reverses the bit order, and combines them into a 32-bit word per bit position.
    
    The final result concatenates the processed bit words for all sub-vectors.
    
    Parameters:
    - vector: Input vector to be preprocessed.
    - dtype: Data type of the input vector. Supported: np.int8 and np.int32.
    - name: Name used in the printed C-style array declaration.
    
    Returns:
    - result: Processed vector as a 1D numpy array of 32-bit unsigned integers,
              containing the concatenated bit words for each sub-vector.
    """
    # 检查向量长度是否为32的倍数
    if vector.shape[0] % 32 != 0:
        raise ValueError("向量长度必须为32的倍数")
    
    num_chunks = vector.shape[0] // 32
    
    # 根据数据类型设置每个元素处理的位数及转换方式
    if vector.dtype == np.int8:
        bit_count = 8
        convert_func = lambda x: x.astype(np.uint8)
    elif vector.dtype == np.int32:
        bit_count = 32
        convert_func = lambda x: x.astype(np.uint32)
    else:
        raise ValueError("仅支持 np.int8 和 np.int32 类型")
    
    processed_chunks = []  # 用于保存每个32维子向量处理后的结果
    
    # 遍历每个32维的子向量
    for chunk_index in range(num_chunks):
        chunk = vector[chunk_index * 32 : (chunk_index + 1) * 32]
        # 转换为对应的无符号表示
        unsigned_chunk = convert_func(chunk)
        
        # 创建一个保存该子向量每个位处理结果的数组
        chunk_bit_words = np.zeros(bit_count, dtype=np.uint32)
        
        # 对每一位进行处理，生成对应的32位数
        for bit_pos in range(bit_count):
            # 提取每个元素的第bit_pos位
            bits = (unsigned_chunk >> bit_pos) & 0x01
            # 取反（bit-wise NOT后 & 0x01 限制为1位结果）
            bits = ~bits & 0x01
            # 反转顺序（使得最后一个元素的位在最高位， 第一个元素的位在最低位）
            bits = bits[::-1]
            # 将32个bit组合成一个32位的数
            word = 0
            for i in range(32):
                word = (word << 1) | bits[i]
            chunk_bit_words[bit_pos] = word
        
        processed_chunks.append(chunk_bit_words)
    
    # 将所有子向量处理后的结果拼接在一起
    result = np.concatenate(processed_chunks)
    
    # 打印结果，按照C语言数组初始化的格式输出
    print(f"static uint64_t {name} [{result.size}] = {{")
    for i in range(0, result.size, 4):  # 每行打印4个数
        line = ", ".join(f"0x{word:08X}" for word in result[i:i+4])
        if i + 4 < result.size:
            print("    " + line + ",")
        else:
            print("    " + line)
    print("};")
    
    return result

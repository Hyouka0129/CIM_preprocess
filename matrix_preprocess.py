import numpy as np

def matrix_preprocess(matrix, name):
    # 检查矩阵形状是否适合分割为32x32的子矩阵
    height, width = matrix.shape
    if height % 32 != 0 or width % 32 != 0:
        raise ValueError("矩阵的高和宽必须均为32的倍数才能分割成32x32的子矩阵")
    
    # 分割矩阵为32x32的子矩阵，并按顺序扁平化后拼接在一起
    flattened_parts = []
    for i in range(0, width, 32):
        for j in range(0, height, 32):
            submatrix = matrix[j:j+32, i:i+32]
            flattened_parts.append(submatrix.flatten())
    # 将所有子矩阵的扁平化结果连接起来
    flattened = np.concatenate(flattened_parts)
    # 转换为无符号8-bit表示（便于位操作）
    unsigned_data = flattened.astype(np.uint8)
    
    # 检查元素数量是否为8的倍数
    if len(unsigned_data) % 8 != 0:
        raise ValueError("数据元素个数不是8的倍数")
    
    # 将每8个8-bit数合并为一个64-bit数
    packed_data = np.zeros(len(unsigned_data) // 8, dtype=np.uint64)
    for i in range(0, len(unsigned_data), 8):
        bytes_chunk = unsigned_data[i:i+8]
        word = 0
        for j, byte in enumerate(bytes_chunk):
            word |= (np.uint64(byte) << (8 * j))
        packed_data[i//8] = word
    
    # 打印结果，按照C语言数组初始化的格式输出
    print(f"static uint64_t {name} [{int(len(unsigned_data)/8)}] = {{")
    for i in range(0, len(packed_data), 4):  # 每行打印4个数
        line = ", ".join(f"0x{word:016X}" for word in packed_data[i:i+4])
        print("   " + (line + "," if i+4 < len(packed_data) else line))
    print("};")
    
    return packed_data

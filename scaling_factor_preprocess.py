import numpy as np

# 假设原始向量长度是 32 的倍数
def scaling_factor_preprocess(scaling_factor, name):
    # 拆成多个 32维向量
    chunks = np.split(scaling_factor, len(scaling_factor) // 32)

    # 每个变成 32x32 的对角阵 → flatten
    flat_diag_blocks = [np.diag(chunk).flatten() for chunk in chunks]

    # 拼接
    result_vector = np.concatenate(flat_diag_blocks)

    packed_data = np.zeros(len(result_vector) // 8, dtype=np.uint64)
    for i in range(0, len(result_vector), 8):
        bytes_chunk = result_vector[i:i+8]
        word = 0
        for j, byte in enumerate(bytes_chunk):
            word |= (byte << (8 * j))
        packed_data[i//8] = word
    
    # 打印结果，按照C语言数组初始化的格式输出
    print(f"static uint64_t {name} [{int(len(result_vector)/8)}] = {{")
    for i in range(0, len(packed_data), 4):  # 每行打印4个数
        line = ", ".join(f"0x{word:016X}" for word in packed_data[i:i+4])
        print("   " + (line + "," if i+4 < len(packed_data) else line))
    print("};")
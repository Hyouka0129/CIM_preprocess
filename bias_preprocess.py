import numpy as np

def bias_preprocess (bias,name):
    flattened = bias.flatten()

    # 转换为无符号8-bit表示（便于位操作）
    unsigned_data = flattened.astype(np.uint32)

    # 检查元素数量是否为8的倍数（32x32=1024，1024/8=128）
    assert len(unsigned_data) % 2 == 0

    # 将每8个8-bit数合并为一个64-bit数
    packed_data = np.zeros(len(unsigned_data) // 2, dtype=np.uint64)
    for i in range(0, len(unsigned_data), 2):
        # 获取当前8个字节
        bytes_chunk = unsigned_data[i:i+2]
        # 合并为64-bit数（小端序：第一个字节在最低位）
        word = 0
        for j, byte in enumerate(bytes_chunk):
            word |= (byte << (32 * j))
        packed_data[i//2] = word

    # 打印结果
    print(f"static uint64_t {name} [{int(len(unsigned_data)/2)}] = {{")
    for i in range(0, len(packed_data), 4):  # 每行打印4个数
        line = ", ".join(f"0x{word:016X}" for word in packed_data[i:i+4])
        print("   " + (line + "," if i+4 < len(packed_data) else line))
    print("};")
    return packed_data
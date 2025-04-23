import numpy as np

def zero_preprocess(zero_point, name):
    zero_point_unsigned = zero_point.view(np.uint32)
    print(f"static uint64_t {name} = 0x00000000{zero_point_unsigned:08X};")
    return zero_point

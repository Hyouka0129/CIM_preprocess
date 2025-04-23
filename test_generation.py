import numpy as np
from vector_preprocess import vector_preprocess
from matrix_preprocess import matrix_preprocess
from bias_preprocess import bias_preprocess
from zero_preprocess import zero_preprocess
from scaling_factor_preprocess import scaling_factor_preprocess


np.random.seed(0)
###################################################################### Input Feature

input_feature = np.random.randint(-128, 127, size=32, dtype=np.int8)
vector_preprocess(input_feature, dtype=np.int8, name="input_feature")

###################################################################### Layer 0

## Model Parameters

weight_0 = np.random.randint(-128, 127, size=(32, 128), dtype=np.int8)
matrix_preprocess(weight_0, name="weight_0")

bias_0 = np.random.randint(-2**15, 2**15-1, size=128, dtype=np.int32)
bias_preprocess(bias_0, name="bias_0")

zero_point_0 = np.random.randint(-8192, 8191, dtype=np.int32)
zero_preprocess(zero_point_0, name="zero_point_0")

scaling_factor_0 = np.random.randint(0, 255, size=128, dtype=np.uint8)
scaling_factor_preprocess(scaling_factor_0, name="scaling_factor_0")

## Forward

input_feature_32 = input_feature.astype(np.int32)
weight_0_32 = weight_0.astype(np.int32)
scaling_factor_32 = scaling_factor_0.astype(np.int32)

output_feature_0 = input_feature_32 @ weight_0_32 + bias_0
output_feature_0 = np.maximum(output_feature_0, zero_point_0)
output_feature_0 = output_feature_0 * scaling_factor_32
output_feature_0 = output_feature_0 // 131072 # Left shift

###################################################################### Layer 1

## Model Parameters

weight_1 = np.random.randint(-128, 127, size=(128, 128), dtype=np.int8)
matrix_preprocess(weight_1, name="weight_1")

bias_1 = np.random.randint(-2**15, 2**15-1, size=128, dtype=np.int32)
bias_preprocess(bias_1, name="bias_1")

zero_point_1 = np.random.randint(-8192, 8191, dtype=np.int32)
zero_preprocess(zero_point_1, name="zero_point_1")

scaling_factor_1 = np.random.randint(0, 255, size=128, dtype=np.uint8)
scaling_factor_preprocess(scaling_factor_1, name="scaling_factor_1")

# ## Forward

weight_1_32 = weight_1.astype(np.int32)
scaling_factor_1_32 = scaling_factor_1.astype(np.int32)

output_feature_1 = output_feature_0 @ weight_1_32 + bias_1
output_feature_1 = np.maximum(output_feature_1, zero_point_1)
output_feature_1 = output_feature_1 * scaling_factor_1_32
output_feature_1 = output_feature_1 // 131072 # Left shift

###################################################################### Layer 2

## Model Parameters

weight_2 = np.random.randint(-128, 127, size=(128, 32), dtype=np.int8)
matrix_preprocess(weight_2, name="weight_2")

bias_2 = np.random.randint(-2**15, 2**15-1, size=32, dtype=np.int32)
bias_preprocess(bias_2, name="bias_2")

zero_point_2 = np.random.randint(-8192, 8191, dtype=np.int32)
zero_preprocess(zero_point_2, name="zero_point_2")

scaling_factor_2 = np.random.randint(0, 255, size=32, dtype=np.uint8)
scaling_factor_preprocess(scaling_factor_2, name="scaling_factor_2")

## Forward

weight_2_32 = weight_2.astype(np.int32)
scaling_factor_2_32 = scaling_factor_2.astype(np.int32)

output_feature_2 = output_feature_1 @ weight_2_32 + bias_2
output_feature_2 = np.maximum(output_feature_2, zero_point_2)
output_feature_2 = output_feature_2 * scaling_factor_2_32
output_feature_2 = output_feature_2 // 8192 # Left shift
vector_preprocess(output_feature_2, dtype=np.int32, name="output_feature_2")

# # 打印原始矩阵形状和部分内容用于参考
# vector_unsigned = vector.view(np.uint8)
# result_unsigned = result.view(np.uint32)
# np.savetxt('../data/bias.txt', bias, fmt='%d')
# np.savetxt('../data/result.txt', result_unsigned, fmt='%x')
np.savetxt('../data/matrix.txt', output_feature_1, fmt='%d')
# np.savetxt('../data/vector.txt', vector_unsigned, fmt='%x')
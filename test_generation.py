import numpy as np
from vector_preprocess import vector_preprocess
from matrix_preprocess import matrix_preprocess
from bias_preprocess import bias_preprocess
from zero_preprocess import zero_preprocess
from scaling_factor_preprocess import scaling_factor_preprocess
from input_preprocess import input_preprocess, output_postprocess

np.random.seed(42)

# Define number of iterations and sensors
NUM_ITR = 1
NUM_SENSORS = 14

###################################################################### Input Feature

# Get quantized data for all iterations
quantized_thermal, quantized_power = input_preprocess(NUM_ITR, NUM_SENSORS)

# Process each iteration
for itr in range(NUM_ITR):
    print(f"\n###### Processing Iteration {itr} ######")
    
    # Create input feature by concatenating thermal and power data
    input_feature = np.zeros(32, dtype=np.int8)
    # Fix: use np.concatenate instead of np.concat and select the specific iteration
    input_feature[0:28] = np.concatenate([quantized_thermal[itr], quantized_power[itr]])[:28]
    
    # Output the input feature for this iteration
    vector_preprocess(input_feature, dtype=np.int8, name=f"input_feature_itr{itr}")
    
    ###################################################################### Layer 0
    
    ## Model Parameters
    weight_0_actual = np.random.randint(-128, 127, size=(28, 128), dtype=np.int8)
    weight_0 = np.zeros((32, 128), dtype=np.int8)
    weight_0[0:28, :] = weight_0_actual # Fill the first 28 rows with actual weights
    
    # Only output the weights once since they're the same for all iterations
    if itr == 0:
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
    
    ## Model Parameters - only define once
    if itr == 0:
        weight_1 = np.random.randint(-128, 127, size=(128, 128), dtype=np.int8)
        matrix_preprocess(weight_1, name="weight_1")
        
        bias_1 = np.random.randint(-2**15, 2**15-1, size=128, dtype=np.int32)
        bias_preprocess(bias_1, name="bias_1")
        
        zero_point_1 = np.random.randint(-8192, 8191, dtype=np.int32)
        zero_preprocess(zero_point_1, name="zero_point_1")
        
        scaling_factor_1 = np.random.randint(0, 255, size=128, dtype=np.uint8)
        scaling_factor_preprocess(scaling_factor_1, name="scaling_factor_1")
    
    ## Forward
    weight_1_32 = weight_1.astype(np.int32)
    scaling_factor_1_32 = scaling_factor_1.astype(np.int32)
    
    output_feature_1 = output_feature_0 @ weight_1_32 + bias_1
    output_feature_1 = np.maximum(output_feature_1, zero_point_1)
    output_feature_1 = output_feature_1 * scaling_factor_1_32
    output_feature_1 = output_feature_1 // 131072 # Left shift
    
    ###################################################################### Layer 2
    
    ## Model Parameters - only define once
    if itr == 0:
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
    # output_feature_2 = output_feature_2 * scaling_factor_2_32
    # output_feature_2 = output_feature_2 // 8192 # Left shift
    
    # Output the results for this iteration
    vector_preprocess(output_feature_2, dtype=np.int32, name=f"output_feature_2_itr{itr}")
    
    # Dequantize and print the results
    dequantized_output = output_postprocess(output_feature_2.reshape(1, -1), 1, NUM_SENSORS)
    
    # Print the dequantized output feature as uint16
    print(f"\nDequantized Output Feature (Iteration {itr}):")
    print(dequantized_output.astype(np.uint16))
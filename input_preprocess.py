import numpy as np

def generate_synthetic_sensor_data(
    num_iterations, 
    num_sensors,
    synthetic_thermal_encodings,
    synthetic_voltage_encodings,
    synthetic_current_encodings,
    t_encoding_width=10,
    v_encoding_width=10, 
    i_encoding_width=16,
    p_encoding_width=26
):
    """
    Generate synthetic sensor data similar to the SystemVerilog testbench.
    
    Args:
        num_iterations: Number of iterations
        num_sensors: Number of sensors
        synthetic_thermal_encodings: Base value for thermal encodings
        synthetic_voltage_encodings: Base value for voltage encodings
        synthetic_current_encodings: Base value for current encodings
        t_encoding_width: Bit width for thermal sensor data (default 10)
        v_encoding_width: Bit width for voltage sensor data (default 10)
        i_encoding_width: Bit width for current sensor data (default 16)
        p_encoding_width: Bit width for power sensor data (default 26)
        
    Returns:
        Tuple of numpy arrays (thermal_data, voltage_data, current_data, power_data)
    """
    # Initialize arrays
    thermal_data = np.zeros((num_iterations, num_sensors), dtype=np.uint16)
    voltage_data = np.zeros((num_iterations, num_sensors), dtype=np.uint16)
    current_data = np.zeros((num_iterations, num_sensors), dtype=np.uint32)
    power_data = np.zeros((num_iterations, num_sensors), dtype=np.uint32)
    
    # Calculate max values based on bit widths
    t_max = (1 << t_encoding_width) - 1
    v_max = (1 << v_encoding_width) - 1
    i_max = (1 << i_encoding_width) - 1
    p_max = (1 << p_encoding_width) - 1
    
    # Generate data using the same pattern as the testbench
    for itr in range(num_iterations):
        for j in range(num_sensors):
            # Calculate raw values
            thermal_data[itr][j] = synthetic_thermal_encodings + 1 * j + (itr + 1)
            voltage_data[itr][j] = synthetic_voltage_encodings + 2 * j + (itr + 1)
            current_data[itr][j] = synthetic_current_encodings + 3 * j + (itr + 1)
            
            # Apply bit width constraints (wrap around if exceeding max value)
            thermal_data[itr][j] &= t_max
            voltage_data[itr][j] &= v_max
            current_data[itr][j] &= i_max
            
            # Calculate power
            power_data[itr][j] = voltage_data[itr][j] * current_data[itr][j]
            # Apply bit width constraint to power data
            power_data[itr][j] &= p_max
    
    return thermal_data, voltage_data, current_data, power_data


def quantize_sensor_data(
    raw_data,
    scale_multiplier,
    scale_shift,
    zero_point,
    encoding_width=10,
    quantized_width=8
):
    # Multiply by scale factor
    mult_result = raw_data * scale_multiplier
    
    # Add rounding factor
    rounded = mult_result + (1 << (scale_shift - 1))
    
    # Shift right
    shifted = rounded >> scale_shift
    
    # Add zero point with proper sign extension (matching RTL behavior)
    # First convert unsigned shifted result to signed by adding a leading zero
    shifted_signed = int(shifted)
    
    # Sign-extend zero_point to match width of shifted result
    extended_width = encoding_width + 16  # Matches ENCODING_WIDTH+SCALE_WIDTH
    if zero_point < 0:
        sign_extended_zero_point = zero_point | ((-1) << quantized_width)
    else:
        sign_extended_zero_point = zero_point
        
    # Add with proper widths
    with_zero_point = shifted_signed + sign_extended_zero_point
    
    # Saturate to quantized width with proper bit pattern matching
    max_val = (1 << (quantized_width - 1)) - 1
    min_val = -(1 << (quantized_width - 1))
    
    if with_zero_point > max_val:
        quantized = max_val
    elif with_zero_point < min_val:
        quantized = min_val
    else:
        quantized = with_zero_point
    
    return np.int8(quantized)

def generate_quantized_data(thermal_data, power_data, num_sensors):
    """
    Generate quantized thermal and power data using linear progression for scaling factors.
    
    Args:
        thermal_data: Raw thermal sensor data (array of shape [num_iterations, num_sensors])
        power_data: Raw power sensor data (array of shape [num_iterations, num_sensors])
        num_sensors: Number of sensors
        
    Returns:
        Tuple of (quantized_thermal, quantized_power, config_dict)
    """
    # Arrays to store quantization parameters
    thermal_scale_multiplier = np.zeros(num_sensors, dtype=np.uint16)
    thermal_scale_shift = np.zeros(num_sensors, dtype=np.uint8)
    thermal_zero_point = np.zeros(num_sensors, dtype=np.int8)
    
    power_scale_multiplier = np.zeros(num_sensors, dtype=np.uint16)
    power_scale_shift = np.zeros(num_sensors, dtype=np.uint8)
    power_zero_point = np.zeros(num_sensors, dtype=np.int8)
    
    # Initialize quantization parameters using linear progression
    for i in range(num_sensors):
        # Temperature quantization - linear progression
        thermal_scale_multiplier[i] = 0x0100 + i * 0x0010      # Start at 1.0 and increase by 0.0625
        thermal_scale_shift[i] = 0x08 + (i % 3)                # Shift values: 8, 9, 10 repeating
        thermal_zero_point[i] = np.int8(-64 + i * 8)           # Start at -64 and increase by 8

        # Power quantization - different pattern
        power_scale_multiplier[i] = 0x0020 + i * 0x0008        # Start smaller, increase gradually
        power_scale_shift[i] = 0x06 + (i % 4)                  # Shift values: 6-9 repeating
        power_zero_point[i] = np.int8(-32 + i * 4)             # Start at -32 and increase by 4
    
    # Quantize the data for each sensor and iteration
    num_iterations = thermal_data.shape[0]
    quantized_thermal = np.zeros((num_iterations, num_sensors), dtype=np.int8)
    quantized_power = np.zeros((num_iterations, num_sensors), dtype=np.int8)
    
    for i in range(num_iterations):
        for s in range(num_sensors):
            # Quantize thermal data
            quantized_thermal[i, s] = quantize_sensor_data(
                thermal_data[i, s], 
                thermal_scale_multiplier[s],
                thermal_scale_shift[s],
                thermal_zero_point[s],
                encoding_width=10
            )
            
            # Quantize power data
            quantized_power[i, s] = quantize_sensor_data(
                power_data[i, s],
                power_scale_multiplier[s],
                power_scale_shift[s],
                power_zero_point[s],
                encoding_width=26
            )
    
    # Store config for reference
    config = {
        'thermal_scale_multiplier': thermal_scale_multiplier,
        'thermal_scale_shift': thermal_scale_shift,
        'thermal_zero_point': thermal_zero_point,
        'power_scale_multiplier': power_scale_multiplier,
        'power_scale_shift': power_scale_shift,
        'power_zero_point': power_zero_point
    }
    
    return quantized_thermal, quantized_power, config

def generate_dequantized_data(quantized_data, num_sensors, t_encoding_width=10):
    """
    Generate dequantized data from quantized int32 values (output of matrix multiplication).
    Matches the RTL implementation with zero point subtraction, scale, round, shift, and saturation.
    
    Args:
        quantized_data: Quantized int32 data (array of shape [num_iterations, num_sensors])
        num_sensors: Number of sensors
        t_encoding_width: Target bit width for dequantized values (default 10)
        
    Returns:
        Tuple of (dequantized_data, config_dict) with dequantized values and config parameters
    """
    # Arrays to store dequantization parameters - keeping as uint16 as in RTL
    dequant_scale = np.zeros(num_sensors, dtype=np.uint16)
    dequant_shift = np.zeros(num_sensors, dtype=np.uint8)
    dequant_zero = np.zeros(num_sensors, dtype=np.int32)  # Using int32 for zero point with int32 inputs
    
    # Initialize dequantization parameters using linear progression
    for i in range(num_sensors):
        dequant_scale[i] = 0x0080 + i * 0x0010     # Start at 0.5, increase by 0.0625
        dequant_shift[i] = 0x07 + (i % 3)          # Shifts: 7, 8, 9 repeating
        dequant_zero[i] = np.int32(i * 128)        # Larger zero points for int32 values
    
    # Dequantize the data for each sensor and iteration
    num_iterations = quantized_data.shape[0]
    dequantized_data = np.zeros((num_iterations, num_sensors), dtype=np.uint16)
    
    for i in range(num_iterations):
        for s in range(num_sensors):
            # Get the quantized value as a signed int32
            quant_val = np.int32(quantized_data[i, s])
            
            # Step 1: Subtract zero point
            dequant_zeroed = int(quant_val) - int(dequant_zero[s])
            
            # Step 2: Scale by multiplying with the scale factor
            # Use 64-bit intermediate to avoid overflow with int32 inputs
            dequant_scaled = np.int64(dequant_zeroed) * int(dequant_scale[s])
            
            # Step 3: Add rounding factor
            dequant_rounded = dequant_scaled + (1 << (dequant_shift[s] - 1))
            
            # Step 4: Shift right (arithmetic shift to preserve sign)
            dequant_shifted = dequant_rounded >> dequant_shift[s]
            
            # Step 5: Saturate
            if dequant_shifted < 0:
                # Negative result - saturate to 0
                dequantized_data[i, s] = 0
            elif dequant_shifted > ((1 << t_encoding_width) - 1):
                # Too large - saturate to max value (all ones with t_encoding_width bits)
                dequantized_data[i, s] = (1 << t_encoding_width) - 1
            else:
                # In range - take the value as is
                dequantized_data[i, s] = dequant_shifted
    
    # Store config for reference
    config = {
        'dequant_scale': dequant_scale,
        'dequant_shift': dequant_shift,
        'dequant_zero': dequant_zero
    }
    
    return dequantized_data, config

def input_preprocess(num_itr, num_sensors):
    synthetic_thermal_encodings = 0x10
    synthetic_voltage_encodings = 0x2
    synthetic_current_encodings = 0x3

    thermal_data, voltage_data, current_data, power_data = generate_synthetic_sensor_data(
        num_itr, 
        num_sensors,
        synthetic_thermal_encodings,
        synthetic_voltage_encodings,
        synthetic_current_encodings
    )

    quantized_thermal, quantized_power, config = generate_quantized_data(
        thermal_data, 
        power_data, 
        num_sensors
    )

    # Print example results for quick view (keep this for debugging)
    print("Original vs Quantized Thermal Data (first iteration):")
    for i in range(min(3, num_sensors)):
        print(f"Sensor {i}: Original={thermal_data[0][i]}, Quantized={quantized_thermal[0][i]}")
    
    print("\nOriginal vs Quantized Power Data (first iteration):")
    for i in range(min(3, num_sensors)):
        print(f"Sensor {i}: Original={power_data[0][i]}, Quantized={quantized_power[0][i]}")

    # Save all data to file
    with open('input_feature.txt', 'w') as f:
        f.write("# Input Feature Data\n")
        f.write("# Format: iteration,sensor,thermal_raw,thermal_quantized,power_raw,power_quantized\n\n")
        
        for itr in range(num_itr):
            f.write(f"## Iteration {itr}\n")
            for s in range(num_sensors):
                f.write(f"{itr},{s},{thermal_data[itr][s]},{quantized_thermal[itr][s]},{power_data[itr][s]},{quantized_power[itr][s]}\n")
            f.write("\n")
            
    print(f"\nInput feature data saved to 'input_feature.txt'")

    return quantized_thermal, quantized_power

def output_postprocess(quantized_thermal, num_itr, num_sensors):
    """
    Dequantize the model outputs and save results to file.
    
    Args:
        quantized_thermal: Quantized int32 data from model output
        num_itr: Number of iterations
        num_sensors: Number of sensors
        
    Returns:
        Dequantized data in the original encoding width format
    """
    dequantized_thermal, config = generate_dequantized_data(quantized_thermal, num_sensors)

    # Print some example results for quick debugging
    print("\nDequantized Output Data (first iteration):")
    for i in range(min(3, num_sensors)):
        print(f"Output {i}: Quantized={quantized_thermal[0][i]}, Dequantized={dequantized_thermal[0][i]}")

    # Save all data to file
    with open('output_feature.txt', 'w') as f:
        f.write("# Output Feature Data\n")
        f.write("# Format: iteration,output_index,quantized_value,dequantized_value\n\n")
        
        for itr in range(num_itr):
            f.write(f"## Iteration {itr}\n")
            for s in range(num_sensors):
                f.write(f"{itr},{s},{quantized_thermal[itr][s]},{dequantized_thermal[itr][s]}\n")
            f.write("\n")
            
    print(f"\nOutput feature data saved to 'output_feature.txt'")

    # Generate C array with data from all iterations concatenated
    # Flatten the 2D array to concatenate all iterations
    c_array = dequantized_to_c_array(dequantized_thermal.flatten(), name="dequantized_thermal_data")
    
    print(c_array)

    return dequantized_thermal


def dequantized_to_c_array(dequantized_data, name="dequantized_thermal_data"):
    """
    Convert dequantized thermal data to a C uint64_t array string.
    Each uint16 value is stored in a separate uint64_t.
    
    Args:
        dequantized_data: Numpy array of uint16 dequantized values
        name: Name for the C array
        
    Returns:
        String containing C code for const uint64_t array
    """
    # Ensure data is flattened
    flat_data = dequantized_data.flatten()
    
    # Create one uint64_t for each uint16 value
    num_uint64 = len(flat_data)
    packed_data = np.zeros(num_uint64, dtype=np.uint64)
    
    # Simply copy each value (no bit shifting/packing)
    for i in range(num_uint64):
        packed_data[i] = int(flat_data[i])
    
    # Generate C code
    c_code = f"const uint64_t {name}[{len(packed_data)}] = {{\n"
    for i in range(0, len(packed_data), 4):  # 4 values per line
        line = ", ".join(f"0x{val:016X}" for val in packed_data[i:i+4])
        c_code += f"    {line}"
        if i + 4 < len(packed_data):
            c_code += ",\n"
        else:
            c_code += "\n"
    c_code += "};\n"
    
    return c_code
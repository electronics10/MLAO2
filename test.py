import numpy as np

a = np.array([1,0,1,0,1,0,1,0,1,0,1,1])

def encode_B2T(binary_array):
    decimal_value = int(''.join(map(str, binary_array)), 2)
    hex_str = hex(decimal_value)
    return hex_str

def decode_T2B(topology):
    return np.array([int(bit) for bit in format(int(topology, 16), f'0{12}b')])

b = encode_B2T(a)
print(b)

c = decode_T2B(b)
print(c)
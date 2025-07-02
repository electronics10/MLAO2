import numpy as np
from settings import*
import CST_Controller as cstc

def encode_B2T(binary_array):
    decimal_value = int(''.join(map(str, binary_array)), 2)
    hex_str = hex(decimal_value)
    return hex_str

def decode_T2B(topology):
    return np.array([int(bit) for bit in format(int(topology, 16), f'0{NX*NY}b')])

if __name__ == "__main__":
    binary_array = decode_T2B(VERIFICATION)
    antenna = cstc.CSTInterface(FILEPATH)
    antenna.update_distribution(binary_array)
    antenna.save()

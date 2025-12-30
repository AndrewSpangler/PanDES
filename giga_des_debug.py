#!/usr/bin/env python3
import sys
from direct.showbase.ShowBase import ShowBase
from gigades import GigaDES
from Crypto.Cipher import DES

def str_to_key_py(key_val):
    """Python implementation of the C str_to_key logic for PyCryptodome."""
    # Convert input to 8 bytes (though we only use the first 7 as per C code)
    b = key_val.to_bytes(8, 'big')
    
    key = [0] * 8
    key[0] = (b[0] >> 1)
    key[1] = ((b[0] & 0x01) << 6) | (b[1] >> 2)
    key[2] = ((b[1] & 0x03) << 5) | (b[2] >> 3)
    key[3] = ((b[2] & 0x07) << 4) | (b[3] >> 4)
    key[4] = ((b[3] & 0x0F) << 3) | (b[4] >> 5)
    key[5] = ((b[4] & 0x1F) << 2) | (b[5] >> 6)
    key[6] = ((b[5] & 0x3F) << 1) | (b[6] >> 7)
    key[7] = (b[6] & 0x7F)
    
    # Final (key[i] << 1) shift
    return bytes([(x << 1) & 0xFF for x in key])

class TestApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.disableMouse()
        
def debug_gigades():
    print("Debugging GigaDES with internal str_to_key transformation...")
    app = TestApp()
    gdes = GigaDES(app)
    block = b'kgs!@#$%'
    
    test_keys = [
        0x0000000000000001,
        0x0123456789ABCDEF,
        0x4142434445464700, # Example: ASCII 'ABCDEFG'
    ]
    
    for key_val in test_keys:
        print(f"Testing key 0x{key_val:016X}:")
        
        # CPU path: Transform key first
        cpu_key = str_to_key_py(key_val)
        cipher = DES.new(cpu_key, DES.MODE_ECB)
        cpu_output = cipher.encrypt(block)
        print(f"  CPU (Effective Key {cpu_key.hex()}): {cpu_output.hex()}")
        
        # GPU path: Send raw key (Shader handles transformation)
        result = gdes.process_block(block, 1, key_val)
        gpu_output = result[0:8].tobytes()
        print(f"  GPU: {gpu_output.hex()}")
        
        if cpu_output == gpu_output:
            print("  ✓ MATCH")
        else:
            print("  ✗ MISMATCH")
        print()

if __name__ == "__main__":
    debug_gigades()

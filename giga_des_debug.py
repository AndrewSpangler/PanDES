import sys
from direct.showbase.ShowBase import ShowBase
from gigades import GigaDES
from Crypto.Cipher import DES

def str_to_key_py(key_val):
    """Python implementation of the C str_to_key logic for PyCryptodome."""
    # Convert input to 8-byte integer if needed
    if isinstance(key_val, str):
        # Convert string to bytes, pad/truncate to 8 bytes
        key_bytes = key_val.encode('utf-8')[:8].ljust(8, b'\x00')
        key_val = int.from_bytes(key_bytes, 'big')
    elif isinstance(key_val, bytes):
        # Convert bytes to integer
        key_bytes = key_val[:8].ljust(8, b'\x00')
        key_val = int.from_bytes(key_bytes, 'big')
    
    # Now key_val is an integer, convert to bytes for processing
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
    return bytes([(x << 1) & 0xFF for x in key])

def key_to_int(key_val):
    """Convert various key formats to integer."""
    if isinstance(key_val, str):
        # Convert string to bytes, pad/truncate to 8 bytes, then to int
        key_bytes = key_val.encode('utf-8')[:8].ljust(8, b'\x00')
        return int.from_bytes(key_bytes, 'big')
    elif isinstance(key_val, bytes):
        # Convert bytes to integer
        key_bytes = key_val[:8].ljust(8, b'\x00')
        return int.from_bytes(key_bytes, 'big')
    elif isinstance(key_val, int):
        return key_val
    else:
        raise TypeError(f"Unsupported key type: {type(key_val)}")

class TestApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.disableMouse()
def debug_gigades():
    print("Debugging GigaDES with internal str_to_key transformation...")
    app = TestApp()
    gdes = GigaDES(app)
    
    block = b'KGS!@#$%'
    
    # Keys over 7 chars will be concatenated
    test_keys = {
        'PASSWOR'           : 0xe52cac67419a9a22, # P A S S W O R
        b'PASSWOR'          : 0xe52cac67419a9a22, # P A S S W O R
        'PASSWORD        '  : 0xe52cac67419a9a22, # P A S S W O R
        ''                  : 0xaad3b435b51404ee, # Null
        0x0000000000000000  : 0xaad3b435b51404ee,  # 0\0\0\0\0\0\0
        'D'                 : 0x4a3b108f3fa6cb6d, # D\0\0\0\0\0\0
    }

    
    for key_val, expected_int in test_keys.items():
        # Convert expected int to bytes for comparison
        expected_bytes = expected_int.to_bytes(8, 'big')
        
        # Display the key in a readable format
        if isinstance(key_val, str):
            key_display = f"'{key_val}' (string)"
        elif isinstance(key_val, bytes):
            key_display = f"{key_val} (bytes)"
        else:
            key_display = f"0x{key_val:016X} (int)"
        
        print(f"Testing key {key_display}:")
        
        # Convert to integer for GPU
        key_int = key_to_int(key_val)
        print(f"  Raw key as int: 0x{key_int:016X}")
        
        # CPU path: Transform key first
        cpu_key = str_to_key_py(key_val)
        cipher = DES.new(cpu_key, DES.MODE_ECB)
        cpu_output = cipher.encrypt(block)
        print(f"  CPU (Effective Key {cpu_key.hex()}): {cpu_output.hex()}")
        
        # GPU path: Send raw key (Shader handles transformation)
        result = gdes.process_block(block, 1, key_int)
        gpu_output = result[0:8].tobytes()
        print(f"  GPU: {gpu_output.hex()}")
        
        if cpu_output == gpu_output:
            print("  ✓ CPU and GPU MATCH")
        else:
            raise ValueError("  ✗ CPU and GPU MISMATCH")

        if gpu_output == expected_bytes:
            print(f"  ✓ GPU MATCHES EXPECTED VALUE (0x{expected_int:016x})")
        else:
            print(f"  ✗ GPU DOESN'T MATCH EXPECTED VALUE")
            print(f"    GPU: 0x{gpu_output.hex()}")
            print(f"    Expected: 0x{expected_bytes.hex()}")
            raise ValueError("  ✗ GPU and Expected MISMATCH")
        print()

if __name__ == "__main__":
    debug_gigades()

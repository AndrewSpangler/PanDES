import sys
import binascii
from direct.showbase.ShowBase import ShowBase

from pandes import ParallelDES
from Crypto.Cipher import DES
from Crypto.Util.Padding import pad

class DESTester(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.disableMouse()
        
        print("Initializing GPU DES System...")
        self.pdes = ParallelDES()
        
        # Run the verification tests
        self.run_verification()
        
        # Close the app when done
        print("\nAll tests completed.")
        self.userExit()

    def run_verification(self):
        print(f"\n{'='*70}")
        print(f"STARTING CROSS-VERIFICATION (GPU vs Standard CPU Library)")
        print(f"{'='*70}")

        # 1. Register Keys
        # Note: DES keys are 8 bytes.
        keys = {
            'key1': 0x133457799BBCDFF1,
            'key2': 0xAABBCCDDEEFF0011,
            'key3': 0x0123456789ABCDEF
        }
        
        for k_id, k_val in keys.items():
            self.pdes.add_key(k_id, k_val)

        # 2. Define Test Cases
        test_cases = [
            # (Key ID, Plaintext)
            ('key1', b"Hello World"),             # Standard text
            ('key2', b"12345678"),                # Exactly 8 bytes (1 block)
            ('key3', b""),                        # Empty string (Edge case)
            ('key1', b"Longer message spanning multiple blocks of DES data."),
            ('key2', b"1234567"),                 # 7 bytes (Needs padding)
        ]

        # 3. Prepare Batch Jobs
        gpu_jobs = []
        for i, (k_id, text) in enumerate(test_cases):
            gpu_jobs.append({
                'data': text,
                'key_id': k_id,
                'operation': 'encrypt'
            })

        # 4. EXECUTE GPU ENCRYPTION
        print(f"\n[GPU] Dispatching {len(gpu_jobs)} encryption jobs...")
        gpu_ciphertexts = self.pdes.execute_parallel(gpu_jobs)

        # 5. VERIFY ACCURACY
        failures = 0
        
        print(f"\n{'ID':<4} | {'Result':<8} | {'Comparison Info'}")
        print("-" * 70)

        for i, (k_id, plaintext) in enumerate(test_cases):
            # --- A. Standard CPU Encryption (The "Truth") ---
            key_int = keys[k_id]
            key_bytes = key_int.to_bytes(8, byteorder='big')
            
            cipher = DES.new(key_bytes, DES.MODE_ECB)
            
            # PyCryptodome 'pad' adds PKCS7 padding, just like your class
            # Block size for DES is 8 bytes
            try:
                standard_input = pad(plaintext, 8) if plaintext else pad(b'', 8)
                expected_ciphertext = cipher.encrypt(standard_input)
            except Exception as e:
                expected_ciphertext = b''
                print(f"Standard lib error: {e}")

            # --- B. Compare ---
            actual_ciphertext = gpu_ciphertexts[i]
            
            matches = (actual_ciphertext == expected_ciphertext)
            status = "PASS" if matches else "FAIL"
            
            if not matches:
                failures += 1
                
            # Log specific hex snippets for visual check
            short_hex_gpu = binascii.hexlify(actual_ciphertext[:8]).decode().upper()
            short_hex_cpu = binascii.hexlify(expected_ciphertext[:8]).decode().upper()
            
            print(f"{i:<4} | {status:<8} | GPU: {short_hex_gpu}... vs CPU: {short_hex_cpu}...")

        if failures == 0:
            print(f"\nSUCCESS: All {len(test_cases)} GPU encryptions match standard DES implementation exactly.")
        else:
            print(f"\nFAILURE: {failures} mismatches detected.")
            return

        # 6. VERIFY DECRYPTION ROUND-TRIP
        print(f"\n{'='*70}")
        print("VERIFYING DECRYPTION (Round Trip)")
        print(f"{'='*70}")
        
        decrypt_jobs = []
        for i, ciphertext in enumerate(gpu_ciphertexts):
            decrypt_jobs.append({
                'data': ciphertext,
                'key_id': test_cases[i][0],
                'operation': 'decrypt'
            })
            
        print(f"[GPU] Dispatching {len(decrypt_jobs)} decryption jobs...")
        decrypted_results = self.pdes.execute_parallel(decrypt_jobs)
        
        decrypt_failures = 0
        for i, result in enumerate(decrypted_results):
            original = test_cases[i][1]
            if result != original:
                decrypt_failures += 1
                print(f"Job {i} Failed: Got {result} expected {original}")
        
        if decrypt_failures == 0:
            print(f"SUCCESS: All {len(decrypt_jobs)} messages successfully decrypted back to original plaintext.")
        else:
            print(f"FAILURE: {decrypt_failures} decryption errors.")

if __name__ == "__main__":
    app = DESTester()
    app.run()

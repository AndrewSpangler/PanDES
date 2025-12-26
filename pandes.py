import numpy as np
from array import array
from panda3d.core import NodePath, Shader, ShaderAttrib, ShaderBuffer, GeomEnums, ComputeNode
from direct.showbase.ShowBase import ShowBase

from des_constants import (
    PC1, PC2, SHIFTS, S1, S2, S3, S4, S5, S6, S7, S8, IP, FP, E, P
)

def tostr(items: list) -> str:
    return ", ".join(str(i) for i in items)


class ParallelDES:
    DES_SHADER = f"""#version 430
    layout (local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;
    
    // Structure to hold a DES job
    struct DESJob {{
        uint startBlock;      // Starting block index in input buffer
        uint numBlocks;       // Number of blocks to process
        uint keyIndex;        // Which key to use (offset in key buffer)
        uint operation;       // 0 = encrypt, 1 = decrypt
        uint outputOffset;    // Where to write results
        uint padding[3];      // Padding for alignment
    }};
    
    // Input/output buffers
    layout(std430) buffer InputData {{ uint inputBlocks[]; }};
    layout(std430) buffer OutputData {{ uint outputBlocks[]; }};
    layout(std430) buffer KeyData {{ uint roundKeys[]; }};
    layout(std430) buffer JobData {{ DESJob jobs[]; }};
    
    uniform int numJobs;
    
    // DES permutation tables
    const int IP[64] = int[64]({tostr(IP)});
    const int FP[64] = int[64]({tostr(FP)});
    const int E[48] = int[48]({tostr(E)});
    const int P[32] = int[32]({tostr(P)});
    
    // S-boxes
    const int S1[64] = int[64]({tostr(S1)});
    const int S2[64] = int[64]({tostr(S2)});
    const int S3[64] = int[64]({tostr(S3)});
    const int S4[64] = int[64]({tostr(S4)});
    const int S5[64] = int[64]({tostr(S5)});
    const int S6[64] = int[64]({tostr(S6)});
    const int S7[64] = int[64]({tostr(S7)});
    const int S8[64] = int[64]({tostr(S8)});
    
    // Helper functions
    uint getBit(uint value, int pos) {{
        return (value >> (32 - pos)) & 1u;
    }}
    
    uint getBit64(uint high, uint low, int pos) {{
        if (pos <= 32) {{
            return (high >> (32 - pos)) & 1u;
        }} else {{
            return (low >> (64 - pos)) & 1u;
        }}
    }}
    
    uint setBit(uint value, int pos, uint bit) {{
        uint mask = 1u << (32 - pos);
        if (bit != 0u) {{
            return value | mask;
        }} else {{
            return value & ~mask;
        }}
    }}
    
    void permute64(uint in_high, uint in_low, const int table[64], out uint out_high, out uint out_low) {{
        out_high = 0u;
        out_low = 0u;
        
        for (int i = 0; i < 32; i++) {{
            uint bit = getBit64(in_high, in_low, table[i]);
            out_high = setBit(out_high, i + 1, bit);
        }}
        
        for (int i = 32; i < 64; i++) {{
            uint bit = getBit64(in_high, in_low, table[i]);
            out_low = setBit(out_low, i - 31, bit);
        }}
    }}
    
    uint[2] expand(uint r) {{
        uint result[2];
        result[0] = 0u;
        result[1] = 0u;
        
        for (int i = 0; i < 24; i++) {{
            uint bit = getBit(r, E[i]);
            result[0] = setBit(result[0], i + 1, bit);
        }}
        
        for (int i = 24; i < 48; i++) {{
            uint bit = getBit(r, E[i]);
            result[1] = setBit(result[1], i - 23, bit);
        }}
        
        return result;
    }}
    
    uint sbox(int box_num, uint input) {{
        const int sboxes[8][64] = int[8][64](S1, S2, S3, S4, S5, S6, S7, S8);
        
        uint row = ((input & 0x20u) >> 4) | (input & 0x01u);
        uint col = (input & 0x1Eu) >> 1;
        
        int index = int(row * 16u + col);
        return uint(sboxes[box_num][index]);
    }}
    
    uint feistel(uint r, uint subkey_high, uint subkey_low) {{
        uint expanded[2] = expand(r);
        
        expanded[0] ^= subkey_high;
        expanded[1] ^= subkey_low;
        
        uint result = 0u;
        
        for (int i = 0; i < 8; i++) {{
            uint input_bits;
            if (i < 4) {{
                input_bits = (expanded[0] >> (26 - i * 6)) & 0x3Fu;
            }} else {{
                input_bits = (expanded[1] >> (26 - (i - 4) * 6)) & 0x3Fu;
            }}
            
            uint sbox_out = sbox(i, input_bits);
            result |= (sbox_out << (28 - i * 4));
        }}
        
        uint permuted = 0u;
        for (int i = 0; i < 32; i++) {{
            uint bit = getBit(result, P[i]);
            permuted = setBit(permuted, i + 1, bit);
        }}
        
        return permuted;
    }}
    
    void des_process(uint in_high, uint in_low, uint keyOffset, uint operation, out uint out_high, out uint out_low) {{
        uint ip_high, ip_low;
        permute64(in_high, in_low, IP, ip_high, ip_low);
        
        uint left = ip_high;
        uint right = ip_low;
        
        for (int round = 0; round < 16; round++) {{
            int key_idx;
            if (operation == 0u) {{
                key_idx = int(keyOffset) + round * 2;
            }} else {{
                key_idx = int(keyOffset) + (15 - round) * 2;
            }}
            
            uint subkey_high = roundKeys[key_idx];
            uint subkey_low = roundKeys[key_idx + 1];
            
            uint temp = right;
            right = left ^ feistel(right, subkey_high, subkey_low);
            left = temp;
        }}
        
        uint pre_fp_high = right;
        uint pre_fp_low = left;
        
        permute64(pre_fp_high, pre_fp_low, FP, out_high, out_low);
    }}
    
    void main() {{
        uint jobId = gl_WorkGroupID.x;
        uint blockInJob = gl_LocalInvocationID.x;
        
        if (jobId >= uint(numJobs)) return;
        
        DESJob job = jobs[jobId];
        
        if (blockInJob >= job.numBlocks) return;
        
        // Calculate global block index
        uint globalBlockIdx = job.startBlock + blockInJob;
        uint inputIdx = globalBlockIdx * 2u;
        
        // Read input block
        uint in_high = inputBlocks[inputIdx];
        uint in_low = inputBlocks[inputIdx + 1u];
        
        // Process with DES
        uint out_high, out_low;
        des_process(in_high, in_low, job.keyIndex, job.operation, out_high, out_low);
        
        // Write to output
        uint outputIdx = (job.outputOffset + blockInJob) * 2u;
        outputBlocks[outputIdx] = out_high;
        outputBlocks[outputIdx + 1u] = out_low;
    }}
    """
    
    def __init__(self):
        """Initialize parallel DES system"""
        print("Initializing Parallel DES...")
        
        # Create compute shader
        self.shader = Shader.make_compute(Shader.SL_GLSL, self.DES_SHADER)
        
        # Create compute node
        self.compute_node = ComputeNode("parallel_des")
        self.compute_node.add_dispatch(1, 1, 1)  # Will be updated per dispatch
        
        self.np = NodePath(self.compute_node)
        self.np.set_shader(self.shader)
        
        # Storage for keys
        self.keys = {}  # key_id -> round_keys
        self.key_buffer = None
        
        print("Parallel DES initialized")
    
    def _permute_bits(self, value, table, input_bits=64):
        """Apply a permutation table to a value"""
        result = 0
        for i, pos in enumerate(table):
            if value & (1 << (input_bits - pos)):
                result |= (1 << (len(table) - 1 - i))
        return result
    
    def _left_rotate(self, value, shifts, bits=28):
        """Left circular shift"""
        return ((value << shifts) | (value >> (bits - shifts))) & ((1 << bits) - 1)
    
    def generate_subkeys(self, key):
        """Generate 16 subkeys from a 64-bit DES key"""
        if isinstance(key, bytes):
            key = int.from_bytes(key, byteorder='big')
        
        # Apply PC1 permutation
        permuted_key = self._permute_bits(key, PC1, 64)
        
        # Split into C and D halves
        c = (permuted_key >> 28) & 0xFFFFFFF
        d = permuted_key & 0xFFFFFFF
        
        round_keys = []
        
        for i in range(16):
            c = self._left_rotate(c, SHIFTS[i], 28)
            d = self._left_rotate(d, SHIFTS[i], 28)
            
            cd = (c << 28) | d
            round_key = self._permute_bits(cd, PC2, 56)
            
            
            high_part = ((round_key >> 24) & 0xFFFFFF) << 8
            low_part = (round_key & 0xFFFFFF) << 8
            
            round_keys.append(high_part)
            round_keys.append(low_part)
        
        return round_keys
    
    def add_key(self, key_id, key):
        """Add a key to the key storage"""
        print(f"Adding key {key_id}")
        subkeys = self.generate_subkeys(key)
        self.keys[key_id] = subkeys
        return len(self.keys) - 1  # Return index
    
    def _build_key_buffer(self):
        """Build unified key buffer from all registered keys"""
        all_keys = []
        for key_id in sorted(self.keys.keys()):
            all_keys.extend(self.keys[key_id])
        
        key_data = array('I', all_keys)
        key_bytes = key_data.tobytes()
        
        self.key_buffer = ShaderBuffer("KeyData", key_bytes, GeomEnums.UH_static)
        self.np.set_shader_input("KeyData", self.key_buffer)
        
        print(f"Built key buffer with {len(all_keys)} uint32 values for {len(self.keys)} keys")
    
    def setup_parallel_jobs(self, jobs):
        """
        Setup multiple DES jobs to run in parallel
        
        jobs: List of dicts with keys:
            - 'data': bytes to encrypt/decrypt
            - 'key_id': which key to use
            - 'operation': 'encrypt' or 'decrypt'
        """
        # Build key buffer if not already done
        if self.key_buffer is None:
            self._build_key_buffer()
        
        # Prepare all input data and job descriptors
        all_input_blocks = []
        job_descriptors = []
        output_info = []
        
        current_input_offset = 0
        current_output_offset = 0
        
        for job_idx, job in enumerate(jobs):
            data = job['data']
            key_id = job['key_id']
            operation = 1 if job['operation'] == 'decrypt' else 0
            
            # Pad data
            if isinstance(data, bytes):
                original_length = len(data)
                
                # PKCS#7 padding
                if operation == 0:  # encrypt
                    padding_length = 8 - (len(data) % 8)
                    if padding_length == 0:
                        padding_length = 8
                    data = data + bytes([padding_length] * padding_length)
                
                # Convert to blocks
                blocks = []
                for i in range(0, len(data), 8):
                    block = int.from_bytes(data[i:i+8], byteorder='big')
                    blocks.append(block)
            else:
                blocks = data
                original_length = len(blocks) * 8
            
            num_blocks = len(blocks)
            
            # Add blocks to input buffer
            for block in blocks:
                all_input_blocks.append((block >> 32) & 0xFFFFFFFF)
                all_input_blocks.append(block & 0xFFFFFFFF)
            
            # Get key offset
            key_index = sorted(self.keys.keys()).index(key_id)
            key_offset = key_index * 32  # 32 uint32s per key
            
            # Create job descriptor (8 uint32s for padding/alignment)
            job_desc = [
                current_input_offset,   # startBlock
                num_blocks,             # numBlocks
                key_offset,             # keyIndex
                operation,              # operation
                current_output_offset,  # outputOffset
                0, 0, 0                 # padding
            ]
            job_descriptors.extend(job_desc)
            
            # Track output info
            output_info.append({
                'offset': current_output_offset,
                'num_blocks': num_blocks,
                'original_length': original_length,
                'operation': operation
            })
            
            current_input_offset += num_blocks
            current_output_offset += num_blocks
        
        # Create input buffer
        input_array = array('I', all_input_blocks)
        input_bytes = input_array.tobytes()
        
        input_buffer = ShaderBuffer("InputData", input_bytes, GeomEnums.UH_stream)
        self.np.set_shader_input("InputData", input_buffer)
        
        # Create output buffer
        output_array = array('I', [0] * len(all_input_blocks))
        output_bytes = output_array.tobytes()
        
        output_buffer = ShaderBuffer("OutputData", output_bytes, GeomEnums.UH_stream)
        self.np.set_shader_input("OutputData", output_buffer)
        
        # Create job buffer
        job_array = array('I', job_descriptors)
        job_bytes = job_array.tobytes()
        
        job_buffer = ShaderBuffer("JobData", job_bytes, GeomEnums.UH_static)
        self.np.set_shader_input("JobData", job_buffer)
        
        # Set uniforms
        self.np.set_shader_input("numJobs", len(jobs))
        
        return output_buffer, output_info, len(jobs)
    
    def execute_parallel(self, jobs):
        """Execute multiple DES operations in parallel"""
        print(f"Executing {len(jobs)} parallel DES jobs")
        
        # Setup jobs
        output_buffer, output_info, num_jobs = self.setup_parallel_jobs(jobs)
        
        # Update compute node dispatch
        # One workgroup per job, 256 threads per workgroup
        self.compute_node.clear_dispatches()
        self.compute_node.add_dispatch(num_jobs, 1, 1)
        
        # Get shader attribute
        sattr = self.np.get_attrib(ShaderAttrib)

        base.graphicsEngine.dispatch_compute(
            (num_jobs, 1, 1),
            sattr,
            base.win.get_gsg()
        )
        
        # Wait for completion
        base.graphicsEngine.sync_frame()
        
        # Extract results
        engine = base.win.gsg.get_engine()
        result_bytes = engine.extract_shader_buffer_data(output_buffer, base.win.gsg)
        
        if not result_bytes:
            raise RuntimeError("Could not extract buffer data")
        
        # Convert to array
        result_array = array('I')
        result_array.frombytes(result_bytes)
        
        # Parse results for each job
        results = []
        for i, info in enumerate(output_info):
            offset = info['offset'] * 2  # 2 uint32s per block
            num_blocks = info['num_blocks']
            
            # Extract blocks for this job
            job_blocks = []
            for b in range(num_blocks):
                idx = offset + b * 2
                high = result_array[idx]
                low = result_array[idx + 1]
                block = (high << 32) | low
                job_blocks.append(block)
            
            # Convert to bytes
            result = b''.join(block.to_bytes(8, byteorder='big') for block in job_blocks)
            
            # Remove PKCS#7 padding for decryption
            if info['operation'] == 1:  # decrypt
                padding_length = result[-1]
                if 1 <= padding_length <= 8:
                    if all(b == padding_length for b in result[-padding_length:]):
                        result = result[:-padding_length]
            
            results.append(result)
            
            # print(f"Job {i}: Processed {num_blocks} blocks, output {len(result)} bytes")
        
        return results

if __name__ == "__main__":
    import time

    base = ShowBase()
    base.disableMouse()
    
    # Create parallel DES instance
    pdes = ParallelDES()
    
    # Register keys
    key1 = 0x133457799BBCDFF1
    key2 = 0xAABBCCDDEEFF0011
    key3 = 0x0123456789ABCDEF
    
    pdes.add_key("key1", key1)
    pdes.add_key("key2", key2)
    pdes.add_key("key3", key3)
        
    NUM_STRESS_TEST = 100000

    # Test with many parallel jobs
    print("\n" + "="*60)
    print(f"STRESS TEST: {NUM_STRESS_TEST} jobs\n")

    stress_jobs = []
    for i in range(NUM_STRESS_TEST):
        key_id = f"key{(i % 3) + 1}"
        data = f"Stress test message {i}: " + "X" * (10 + i % 20)
        stress_jobs.append({
            'data': data.encode(),
            'key_id': key_id,
            'operation': 'encrypt'
        })
    
    start_encryption = time.time()
    stress_results = pdes.execute_parallel(stress_jobs)
    end_encryption = time.time() - start_encryption
    print(f"Encrypted in {end_encryption}")
    
    decrypt_stress = [
        {
            'data': stress_results[i],
            'key_id': stress_jobs[i]['key_id'],
            'operation': 'decrypt'
        }
        for i in range(NUM_STRESS_TEST)
    ]
    
    start_decryption = time.time()
    decrypted_stress = pdes.execute_parallel(decrypt_stress)
    end_decryption = time.time() - start_decryption
    print(f"Dencrypted in {end_decryption}")
    
    all_match = all(
        decrypted_stress[i] == stress_jobs[i]['data']
        for i in range(NUM_STRESS_TEST)
    )
    print(f"All {NUM_STRESS_TEST} messages decrypted correctly: {all_match}")
    
    base.destroy()
    print("\nParallel DES test completed!")


#Sample
"""
base = ShowBase()
base.disableMouse()

pdes = ParallelDES()

key1 = 0x133457799BBCDFF1
key2 = 0xAABBCCDDEEFF0011
key3 = 0x0123456789ABCDEF

pdes.add_key("key1", key1)
pdes.add_key("key2", key2)
pdes.add_key("key3", key3)

jobs = [
    {
        'data': b"Hello from job 1!",
        'key_id': 'key1',
        'operation': 'encrypt'
    },
    {
        'data': b"This is job 2 with a longer message to encrypt",
        'key_id': 'key2',
        'operation': 'encrypt'
    },
    {
        'data': b"Job 3: Quick test",
        'key_id': 'key3',
        'operation': 'encrypt'
    },
    {
        'data': b"Job 4: Another message for parallel processing!",
        'key_id': 'key1',
        'operation': 'encrypt'
    }
]

encrypted_results = pdes.execute_parallel(jobs)

print("\nEncryption Results:")
for i, result in enumerate(encrypted_results):
    print(f"Job {i}: {result.hex()}")

decrypt_jobs = [
    {
        'data': encrypted_results[0],
        'key_id': 'key1',
        'operation': 'decrypt'
    },
    {
        'data': encrypted_results[1],
        'key_id': 'key2',
        'operation': 'decrypt'
    },
    {
        'data': encrypted_results[2],
        'key_id': 'key3',
        'operation': 'decrypt'
    },
    {
        'data': encrypted_results[3],
        'key_id': 'key1',
        'operation': 'decrypt'
    }
]

decrypted_results = pdes.execute_parallel(decrypt_jobs)

print("\nDecryption Results:")
for i, result in enumerate(decrypted_results):
    print(f"Job {i}: {result}")
    original = jobs[i]['data']
    print(f"  Match: {result == original}")
"""

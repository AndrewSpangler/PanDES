import sys
import numpy as np
import time
import struct
from panda3d.core import (
    NodePath, Shader, ShaderAttrib, ShaderBuffer, 
    GeomEnums, ComputeNode, load_prc_file_data,
    GraphicsPipeSelection, FrameBufferProperties, WindowProperties, GraphicsPipe
)

for k, v in {
    "window-type": "none",
    "audio-library-name": "null",
    "notify-level-glgsg": "error",
    "gl-debug": "#f",
    "sync-video": "#f",
}.items():
    load_prc_file_data("", f"{k} {v}")

class GigaDES:
    DES_SHADER = """#version 430
    layout (local_size_x = 64) in;
    layout(std430, binding = 1) buffer OD { uint out_data[]; };
    layout(std430, binding = 4) buffer SP { uint sp_boxes[]; };
    uniform uint nBlocks;
    uniform uint input_high;
    uniform uint input_low;
    uniform uint key_start_high;
    uniform uint key_start_low;

    const int PC1[56] = int[](57,49,41,33,25,17,9,1,58,50,42,34,26,18,10,2,59,51,43,35,27,19,11,3,60,52,44,36,63,55,47,39,31,23,15,7,62,54,46,38,30,22,14,6,61,53,45,37,29,21,13,5,28,20,12,4);
    const int PC2[48] = int[](14,17,11,24,1,5,3,28,15,6,21,10,23,19,12,4,26,8,16,7,27,20,13,2,41,52,31,37,47,55,30,40,51,45,33,48,44,49,39,56,34,53,46,42,50,36,29,32);
    const int IP_TAB[64] = int[](58,50,42,34,26,18,10,2,60,52,44,36,28,20,12,4,62,54,46,38,30,22,14,6,64,56,48,40,32,24,16,8,57,49,41,33,25,17,9,1,59,51,43,35,27,19,11,3,61,53,45,37,29,21,13,5,63,55,47,39,31,23,15,7);
    const int FP_TAB[64] = int[](40,8,48,16,56,24,64,32,39,7,47,15,55,23,63,31,38,6,46,14,54,22,62,30,37,5,45,13,53,21,61,29,36,4,44,12,52,20,60,28,35,3,43,11,51,19,59,27,34,2,42,10,50,18,58,26,33,1,41,9,49,17,57,25);
    const int SHIFTS[16] = int[](1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1);

    uint rotate_left_28(uint val, int shift) {
        return ((val << shift) | (val >> (28 - shift))) & 0x0FFFFFFFu;
    }

    void permute64(uint h, uint l, const int table[64], out uint res_h, out uint res_l) {
        res_h = 0u; res_l = 0u;
        for(int i = 0; i < 64; i++) {
            int p = table[i];
            uint bit = (p <= 32) ? ((h >> (32 - p)) & 1u) : ((l >> (64 - p)) & 1u);
            if(i < 32) res_h |= (bit << (31 - i));
            else res_l |= (bit << (63 - i));
        }
    }

    void str_to_key(uint kh, uint kl, out uint oh, out uint ol) {
        uint b[7];
        b[0] = (kh >> 24) & 0xFFu;
        b[1] = (kh >> 16) & 0xFFu;
        b[2] = (kh >> 8)  & 0xFFu;
        b[3] = kh & 0xFFu;
        b[4] = (kl >> 24) & 0xFFu;
        b[5] = (kl >> 16) & 0xFFu;
        b[6] = (kl >> 8)  & 0xFFu;

        uint k[8];
        k[0] = (b[0] >> 1);
        k[1] = ((b[0] & 0x01u) << 6) | (b[1] >> 2);
        k[2] = ((b[1] & 0x03u) << 5) | (b[2] >> 3);
        k[3] = ((b[2] & 0x07u) << 4) | (b[3] >> 4);
        k[4] = ((b[3] & 0x0Fu) << 3) | (b[4] >> 5);
        k[5] = ((b[4] & 0x1Fu) << 2) | (b[5] >> 6);
        k[6] = ((b[5] & 0x3Fu) << 1) | (b[6] >> 7);
        k[7] = (b[6] & 0x7Fu);

        oh = ((k[0] << 1) << 24) | ((k[1] << 1) << 16) | ((k[2] << 1) << 8) | (k[3] << 1);
        ol = ((k[4] << 1) << 24) | ((k[5] << 1) << 16) | ((k[6] << 1) << 8) | (k[7] << 1);
    }

    void permute_pc1(uint h, uint l, out uint C, out uint D) {
        C = 0u; D = 0u;
        for(int i = 0; i < 28; i++) {
            int p = PC1[i];
            uint bit = (p <= 32) ? ((h >> (32 - p)) & 1u) : ((l >> (64 - p)) & 1u);
            C |= (bit << (27 - i));
        }
        for(int i = 28; i < 56; i++) {
            int p = PC1[i];
            uint bit = (p <= 32) ? ((h >> (32 - p)) & 1u) : ((l >> (64 - p)) & 1u);
            D |= (bit << (55 - i));
        }
    }

    void permute_pc2(uint c, uint d, out uint sk0, out uint sk1) {
        sk0 = 0u; sk1 = 0u;
        for(int i = 0; i < 24; i++) {
            int p = PC2[i];
            uint bit = (p <= 28) ? ((c >> (28 - p)) & 1u) : ((d >> (56 - p)) & 1u);
            sk0 |= (bit << (23 - i));
        }
        for(int i = 24; i < 48; i++) {
            int p = PC2[i];
            uint bit = (p <= 28) ? ((c >> (28 - p)) & 1u) : ((d >> (56 - p)) & 1u);
            sk1 |= (bit << (47 - i));
        }
        sk0 <<= 8; sk1 <<= 8;
    }

    uint f(uint r, uint sk0, uint sk1) {
        uint res = 0u;
        res ^= sp_boxes[(0 << 6) + (((r << 5) | (r >> 27)) & 0x3Fu) ^ ((sk0 >> 26) & 0x3Fu)];
        res ^= sp_boxes[(1 << 6) + ((r >> 23) & 0x3Fu) ^ ((sk0 >> 20) & 0x3Fu)];
        res ^= sp_boxes[(2 << 6) + ((r >> 19) & 0x3Fu) ^ ((sk0 >> 14) & 0x3Fu)];
        res ^= sp_boxes[(3 << 6) + ((r >> 15) & 0x3Fu) ^ ((sk0 >> 8) & 0x3Fu)];
        res ^= sp_boxes[(4 << 6) + ((r >> 11) & 0x3Fu) ^ ((sk1 >> 26) & 0x3Fu)];
        res ^= sp_boxes[(5 << 6) + ((r >> 7) & 0x3Fu) ^ ((sk1 >> 20) & 0x3Fu)];
        res ^= sp_boxes[(6 << 6) + ((r >> 3) & 0x3Fu) ^ ((sk1 >> 14) & 0x3Fu)];
        res ^= sp_boxes[(7 << 6) + (((r >> 31) | (r << 1)) & 0x3Fu) ^ ((sk1 >> 8) & 0x3Fu)];
        return res;
    }

    void main() {
        uint bid = gl_GlobalInvocationID.x;
        if(bid >= nBlocks) return;
        
        uint k_low = key_start_low + bid;
        uint k_high = key_start_high + (k_low < key_start_low ? 1u : 0u);
        
        uint tk_high, tk_low;
        str_to_key(k_high, k_low, tk_high, tk_low);

        uint C, D;
        permute_pc1(tk_high, tk_low, C, D);
        
        uint h, l;
        permute64(input_high, input_low, IP_TAB, h, l);
        
        for(int r = 0; r < 16; r++) {
            C = rotate_left_28(C, SHIFTS[r]);
            D = rotate_left_28(D, SHIFTS[r]);
            uint sk0, sk1;
            permute_pc2(C, D, sk0, sk1);
            
            uint t = l;
            l = h ^ f(l, sk0, sk1);
            h = t;
        }
        
        uint res_h, res_l;
        permute64(l, h, FP_TAB, res_h, res_l);
        
        out_data[bid * 2] = ((res_h >> 24) & 0xFFu) | ((res_h >> 8) & 0xFF00u) | 
                            ((res_h << 8) & 0xFF0000u) | ((res_h << 24) & 0xFF000000u);
        out_data[bid * 2 + 1] = ((res_l >> 24) & 0xFFu) | ((res_l >> 8) & 0xFF00u) | 
                                ((res_l << 8) & 0xFF0000u) | ((res_l << 24) & 0xFF000000u);
    }
    """

    def __init__(self, app):
        self.app = app
        pipe = GraphicsPipeSelection.get_global_ptr().make_default_pipe()
        fb_prop = FrameBufferProperties()
        fb_prop.set_rgba_bits(8, 8, 8, 8)
        win_prop = WindowProperties.size(1, 1)
        self.app.win = self.app.graphics_engine.make_output(
            pipe, "headless", 0, fb_prop, win_prop, GraphicsPipe.BF_refuse_window
        )
        self.shader = Shader.make_compute(Shader.SL_GLSL, self.DES_SHADER)
        self.np = NodePath(ComputeNode("giga_des"))
        self.np.set_shader(self.shader)
        self.sp_buf = ShaderBuffer("SP", self._gen_sp(), GeomEnums.UH_static)
        self.np.set_shader_input("SP", self.sp_buf)

    def _gen_sp(self):
        from des_tools import generate_sp_boxes
        return generate_sp_boxes().tobytes()

    def _to_signed(self, n):
        return struct.unpack('i', struct.pack('I', n & 0xFFFFFFFF))[0]

    def process_block(self, block, n_operations, key_start=0x0000000000000001):
        if len(block) != 8: raise ValueError("Block must be 8 bytes")
        b_int = int.from_bytes(block, 'big')
        k_high, k_low = (key_start >> 32) & 0xFFFFFFFF, key_start & 0xFFFFFFFF
        
        self.obuf = ShaderBuffer("OD", n_operations * 8, GeomEnums.UH_stream)
        gsg = self.app.win.get_gsg()
        self.np.set_shader_input("OD", self.obuf)
        self.np.set_shader_input("nBlocks", self._to_signed(n_operations))
        self.np.set_shader_input("input_high", self._to_signed(b_int >> 32))
        self.np.set_shader_input("input_low", self._to_signed(b_int & 0xFFFFFFFF))
        self.np.set_shader_input("key_start_high", self._to_signed(k_high))
        self.np.set_shader_input("key_start_low", self._to_signed(k_low))
        
        self.app.graphics_engine.dispatch_compute(
            ((n_operations + 63) // 64, 1, 1), 
            self.np.get_attrib(ShaderAttrib), gsg
        )
        raw_res = self.app.graphics_engine.extract_shader_buffer_data(self.obuf, gsg)
        return np.frombuffer(raw_res, dtype=np.uint8)

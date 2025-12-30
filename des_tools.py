import numpy as np
from des_constants import *

def generate_sp_boxes():
    def permute(v, table, length):
        res = 0
        for i, p in enumerate(table):
            if v & (1 << (length - p)):
                res |= (1 << (len(table) - 1 - i))
        return res
    sp = np.zeros((8, 64), dtype=np.uint32)
    s_boxes = [S1, S2, S3, S4, S5, S6, S7, S8]
    for i in range(8):
        for val in range(64):
            row = ((val & 0x20) >> 4) | (val & 0x01)
            col = (val >> 1) & 0x0F
            s_out = s_boxes[i][row * 16 + col]
            shifted_s = s_out << (28 - i * 4)
            sp[i][val] = permute(shifted_s, P, 32)
    return sp.flatten()

def generate_subkeys(k):
    def _perm(v, t, b):
        r=0
        for i,p in enumerate(t):
            if v&(1<<(b-p)): r|=1<<(len(t)-1-i)
        return r
    def _rot(v,s,b): return ((v<<s)|(v>>(b-s)))&((1<<b)-1)
    if isinstance(k, bytes): k = int.from_bytes(k, 'big')
    pk = _perm(k, PC1, 64)
    c, d = (pk >> 28) & 0xFFFFFFF, pk & 0xFFFFFFF
    rk = []
    for i in range(16):
        c, d = _rot(c, SHIFTS[i], 28), _rot(d, SHIFTS[i], 28)
        r = _perm((c << 28) | d, PC2, 56)
        rk.extend([((r >> 24) & 0xFFFFFF) << 8, (r & 0xFFFFFF) << 8])
    return rk

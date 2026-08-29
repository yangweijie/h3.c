#!/usr/bin/env python3
"""生成 h3_lora_apply 数值验证数据。

输出:
  tmp_lora_test/lora_test.safetensors  合成 LoRA（BF16，alpha=64, rank=128）
  tmp_lora_test/w_{qkv,out,fc1,fc2}_init.bin     初始权重 BF16（行主序）
  tmp_lora_test/w_{qkv,out,fc1,fc2}_expected.bin 参考合并结果 BF16

h3_lora_apply 合并后应与 expected 匹配（BF16 线性噪声容差内）。
"""
import json, struct
import numpy as np

def to_bf16(arr):
    a = np.float32(arr)
    b = a.view(np.uint32)
    b = (b + np.uint32(0x7fff) + ((b >> np.uint32(16)) & np.uint32(1))).astype(np.uint32)
    b &= np.uint32(0xffff0000)
    return (b >> np.uint32(16)).astype(np.uint16)

def from_bf16(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)

def write_bf16(path, arr):
    with open(path, 'wb') as f:
        f.write(to_bf16(arr).astype('<u2').tobytes())

RANK = 128
ALPHA = 64
scale = ALPHA / RANK
HIDDEN, INNER, FFN = 5376, 7168, 14336

rng = np.random.default_rng(2026)

# 初始权重（未合并）
W_qkv = rng.standard_normal((3*INNER, HIDDEN)).astype(np.float32) * 0.02
W_out = rng.standard_normal((HIDDEN, INNER)).astype(np.float32) * 0.02
W_fc1 = rng.standard_normal((FFN*2, HIDDEN)).astype(np.float32) * 0.02
W_fc2 = rng.standard_normal((HIDDEN, FFN)).astype(np.float32) * 0.02
init = dict(qkv=W_qkv.copy(), out=W_out.copy(), fc1=W_fc1.copy(), fc2=W_fc2.copy())

def lora_factors(rows, in_dim):
    A = (rng.standard_normal((RANK, in_dim)) * 0.02).astype(np.float32)
    B = (rng.standard_normal((rows, RANK)) * 0.02).astype(np.float32)
    return A, B

def merge(W, A, B):
    # F64 累积参考，然后 round 到 F32（近似 BF16 期望）
    delta = (B.astype(np.float64) @ A.astype(np.float64)) * scale
    return W + delta.astype(np.float32)

tensors = {}
# qkv 三带
for key, row0, rows, in_dim in [
        ("attn.to_q", 0, INNER, HIDDEN),
        ("attn.to_k", INNER, INNER, HIDDEN),
        ("attn.to_v", 2*INNER, INNER, HIDDEN)]:
    A, B = lora_factors(rows, in_dim)
    tensors[f"transformer_blocks.0.{key}.lora_A.default.weight"] = A
    tensors[f"transformer_blocks.0.{key}.lora_B.default.weight"] = B
    W_qkv[row0:row0+rows] = merge(W_qkv[row0:row0+rows], A, B)
A, B = lora_factors(HIDDEN, INNER)
tensors["transformer_blocks.0.attn.to_out.0.lora_A.default.weight"] = A
tensors["transformer_blocks.0.attn.to_out.0.lora_B.default.weight"] = B
W_out = merge(W_out, A, B)
A, B = lora_factors(FFN*2, HIDDEN)
tensors["transformer_blocks.0.ff.net.0.proj.lora_A.default.weight"] = A
tensors["transformer_blocks.0.ff.net.0.proj.lora_B.default.weight"] = B
W_fc1 = merge(W_fc1, A, B)
A, B = lora_factors(HIDDEN, FFN)
tensors["transformer_blocks.0.ff.net.2.lora_A.default.weight"] = A
tensors["transformer_blocks.0.ff.net.2.lora_B.default.weight"] = B
W_fc2 = merge(W_fc2, A, B)

# 写出 LoRA safetensors
data = b""
header = {"__metadata__": {"alpha": str(ALPHA), "format": "pt",
                           "key_format": "minimax-h3-diffusers",
                           "floating_dtype": "bfloat16"}}
for name in sorted(tensors):
    arr = tensors[name]
    off = len(data)
    nbytes = int(arr.size) * 2
    header[name] = {"dtype": "BF16", "shape": list(arr.shape),
                    "data_offsets": [off, off + nbytes]}
    data += to_bf16(arr).astype('<u2').tobytes()
hjson = json.dumps(header, separators=(',', ':')).encode()
with open('tmp_lora_test/lora_test.safetensors', 'wb') as f:
    f.write(struct.pack('<Q', len(hjson)))
    f.write(hjson)
    f.write(data)

for tag, merged in [("qkv", W_qkv), ("out", W_out), ("fc1", W_fc1), ("fc2", W_fc2)]:
    write_bf16(f'tmp_lora_test/w_{tag}_init.bin', init[tag])
    write_bf16(f'tmp_lora_test/w_{tag}_expected.bin', merged)
    # 统计 delta 量级
    d = (merged - init[tag]).astype(np.float64)
    print(f"{tag}: shape={merged.shape} delta_rms={np.sqrt((d**2).mean()):.6f} "
          f"w_rms={np.sqrt((merged.astype(np.float64)**2).mean()):.6f}")

print("生成完成: tmp_lora_test/lora_test.safetensors + 4 组 init/expected")

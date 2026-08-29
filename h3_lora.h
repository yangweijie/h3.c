#ifndef H3_LORA_H
#define H3_LORA_H

#include "h3_gpu.h"

#include <stddef.h>

/* Turbo/distillation LoRA merge support.
 *
 * MiniMax-H3 acceleration LoRAs (LightX2V Turbo, official Lightning) are
 * standard safetensors files with BF16 low-rank factors and a Diffusers-style
 * key layout:
 *
 *   transformer_blocks.N.attn.to_q.lora_A.default.weight
 *   transformer_blocks.N.attn.to_q.lora_B.default.weight
 *   transformer_blocks.N.attn.to_k / attn.to_v / attn.to_out.0
 *   transformer_blocks.N.ff.net.0.proj (MLP input) / ff.net.2 (MLP output)
 *   token_refiner.refiner_blocks.N.*       (same sub-keys)
 *
 * Each merged linear becomes W' = W + scale * (B @ A), where A is [r, in],
 * B is [out, r], and scale = alpha / r (the metadata block carries alpha).
 *
 * h3.c consumes the official BF16 checkpoint (model.language_model.layers.N.*
 * for text, blocks.N.* for the DiT) and never rewrites those files; the merge
 * is applied in unified memory at load time so inference still uses plain BF16
 * matrics. SSD streaming and int8 quantization paths do not support merging.
 */

typedef struct h3_lora h3_lora;

/* Open and validate a BF16 LoRA safetensors file. Returns NULL with error set
 * if the file is missing, not BF16, or has no alpha/rank metadata. */
h3_lora *h3_lora_open(const char *path, char *error, size_t error_size);

void h3_lora_close(h3_lora *lora);

/* Merge into the already-loaded GPU tensor `weight`, which represents a
 * [rows_total, in_dim] BF16 matrix laid out row-major.
 *
 *   lora_prefix is "transformer_blocks.N" or "token_refiner.refiner_blocks.N";
 *   target is "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
 *            "ff.net.0.proj", or "ff.net.2".
 *
 * Only the row band [row0, row0 + rows) is updated, so several targets
 * (q/k/v) can share one storage tensor (qkv_proj).
 *
 * The band is replaced by its old value plus the LoRA delta computed on the
 * GPU: delta = scale * (lora_B @ lora_A). Returns 0 on failure. */
int h3_lora_apply(h3_gpu *gpu, h3_lora *lora, h3_gpu_tensor *weight,
                  const char *lora_prefix, const char *target,
                  size_t row0, size_t rows, size_t in_dim,
                  char *error, size_t error_size);

#endif

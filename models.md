# 当前

~~~ yml
fl2va:
  transformer: /Users/jay/h3_sys/MiniMax-H3-Convrot/FL2VA/transformer
  tokenizer:   /Users/jay/h3_sys/MiniMax-H3-Convrot/FL2VA/tokenizer
  # text_encoder 不在 FL2VA/ 内（该目录无 text_encoder 子目录）；
  # 本次运行由 H3_CLIPPROJ_DIR 指向外部 Qwen3-VL-4B 目录充当文本编码器
  text_encoder: /Volumes/data/.lmstudio/models/Qwen3-VL-4B-Instruct-int8-convrot
  video_vae:   /Users/jay/h3_sys/MiniMax-H3-Convrot/FL2VA/video_vae
  audio_vae:   /Users/jay/h3_sys/MiniMax-H3-Convrot/FL2VA/audio_vae
  # ClipProj 投影 (对应 C 的 H3_CLIPPROJ_PROJ)
  clipproj:    /Volumes/data/.lmstudio/models/ClipProj-MiniMax-H3

~~~

组件	文件	磁盘占用	权重 dtype（按体积计）
DiT transformer	FL2VA/transformer/minimax_h3_fastvideo_4step.safetensors	21.33 GiB	I8 主权重 19.74 GiB（250 个大矩阵）+ 配对 F32 scale + U8 zero（*.comfy_quant，行/通道级量化）+ BF16 1.49 GiB（norm/bias/小块） + F16 0.08 GiB + F32 0.02 GiB
time_embedder	FL2VA/transformer/time_embedder.safetensors	0.06 GiB	全 F32（proj_in / proj_out 4 个张量）——已从 DiT 主权重中拆出
text_encoder	qwen3vl_4b_int8_convrot.safetensors（Qwen3-VL-4B-Instruct-int8-convrot）	4.53 GiB	I8 主权重 3.74 GiB（354 个线性矩阵）+ BF16 0.78 GiB（embedding/norm 等）+ F32 weight_scale [out,1] + U8 zero（每输出行 per-channel 量化，354 组），即 int8 行量化 + 已烘焙 convrot 旋转
ClipProj	mmh3-4b-ClipProj-v3-mlp.safetensors	0.47 GiB	全 F16（mean/std、sink_out + 两层 MLP，hidden 32768）
video_vae	video_vae/source/model.safetensors	4.85 GiB	全 F16（560 个张量，VidTok/DiT-VAE 解码器）
audio_vae	audio_vae/model.safetensors	0.56 GiB	全 F32（1087 个张量，DAC + BigVGAN）
tokenizer	tokenizer/	—	纯词表（vocab/merges），无精度概念

# VDN  /Users/jay/.cache/modelscope/models/OpenVDN--vdn-minimax-h3/snapshots/master 
 h3-base/             the released MiniMax-H3: transformer, video and audio VAEs, schedulers · 72 GB
  stage-b-step-2000/   VDN-H3-50-step: linear_branch/ + adapters/default/ LoRA · 4.3 GB
  stage-dmd-step-250/  VDN-H3-8-step: the above + adapters/turbo/ · 5.1 GB


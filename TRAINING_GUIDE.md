# Llama3 200M MoE 预训练方案（MaxText + TinyStories）

## 项目概述

在 Kaggle 上使用 MaxText 预训练 Llama3 200M 参数规模的 MoE 模型（6 层，8 专家，Top‑2 激活），基于 TinyStories 数据集。

## 核心配置

### 模型架构
- **层数**: 6 层
- **MoE 层**: 第 2/4/6 层（每 2 层一个 MoE）
- **专家数**: 8 个专家
- **Top‑K 激活**: 2
- **隐藏层维度** (d_model): 768
- **注意力头数**: 12 个头
- **MLP 维度**: 3072
- **词表大小**: 128,000（Llama3 tokenizer）
- **注意力类型**: MHA（Multi-Head Attention）
- **RoPE**: 默类型（Llama3）
- **权重绑定**: embedding 与 lm_head 共享

### 数据集
- **数据集**: TinyStories（HuggingFace `roneneldanh/TinyStories`）
- **训练集**: `train` split
- **验证集**: `validation` split
- **序列长度**: 2048 tokens
- **Tokenize**: 实时 tokenization

### 训练参数
- **训练轮数**: 2 epochs
- **Batch size**: 4（每设备，保守策略以适应 T4 15GB 显存）
- **精度**: FP16（T4 仅支持 FP16）
- **优化器**: AdamW + Cosine 调度 + Warmup
  - 初始学习率: 3e-5
  - Warmup: 前 10%
  - 最终学习率: 30%
  - 权重衰减: 0.1
- **Checkpoint 频率**: 每 200 步
- **评估频率**: 每个 epoch 一次
- **梯度裁剪**: 1.0
- **梯度累积**: 1 步
- **Dropout**: 0.0（确定性训练）
- **数据 shuffle**: 是（seed=0）

### 硬件
- **平台**: Kaggle Linux
- **GPU**: 2×T4 15GB
- **并行策略**: 数据并行（2 个 GPU）
- **内存策略**: 保守（不启用 gradient checkpointing）

### 日志与导出
- **W&B**:
  - Entity: `gwhwh153td-china-times-org`
  - Project: `moe-llama3-200m`
  - 需要: `WANDB_API_KEY` 环境变量
- **HuggingFace Hub**:
  - 仓库: `lyyh/llama3-200m-moe`
  - 需要: `HF_TOKEN` 环境变量（访问 Llama3 tokenizer 和推送模型）

## 文件结构

```
MOE_200M_maxtext/
├── configs/
│   └── llama3_moe_200m_tinystories.yml    # MaxText 配置文件
├── kaggle_train.sh                           # Kaggle 训练启动脚本
├── export_to_hub.sh                        # HF Hub 导出脚本
└── maxtext/                                  # MaxText 源码（克隆）
    └── src/MaxText/
```

## 使用步骤

### 1. 准备 Kaggle 环境

1. **上传文件到 Kaggle**
   - 将整个项目上传到 Kaggle Notebook 或 Dataset
   - 确保 `maxtext/` 子目录已存在（已克隆）

2. **配置 Kaggle Secrets**
   - 在 Kaggle 设置环境变量（Secrets）：
     - `HF_TOKEN`: 你的 HuggingFace 访问 token（必需）
     - `WANDB_API_KEY`: W&B API key（可选）

3. **验证文件结构**
   ```bash
   ls -la kaggle_train.sh
   ls -la export_to_hub.sh
   ls -la configs/llama3_moe_200m_tinystories.yml
   ```

### 2. 运行训练

在 Kaggle Notebook 中执行：

```bash
chmod +x kaggle_train.sh
./kaggle_train.sh
```

训练脚本会自动：
1. 安装 Python 依赖（JAX、Flax、Optax、Transformers 等）
2. 安装 MaxText（源码编辑模式）
3. 验证环境变量（HF_TOKEN、WANDB_API_KEY）
4. 启动 MaxText 训练
5. 将日志保存到 `training.log`

**预期运行时间**: 约 2-4（取决于 Kaggle 硬件排队）

### 3. 监控训练

查看训练日志：

```bash
tail -f training.log
```

或在 W&B 查看：
- 访问 `https://wandb.ai` → 项目 `gwhwh153td-china-times-org/moe-llama3-200m`
- 查看损失曲线、学习率、训练速度等

### 4. 检查模型

训练完成后，检查点保存在：

```bash
ls -la /kaggle/working/llama3_200m_moe_tinystories/
```

应该看到：
```
drwxr-xr-x  2 admin admin 4096 Jan 1 12:00 checkpoint_000000
drwxr-xr-x  2 admin admin 4096 Jan 1 12:00 checkpoint_000200
...
drwxr-xr-x  2 admin admin 4096 Jan 1 12:00 checkpoint_000600
...
```

### 5. 导出并推送到 HuggingFace Hub

**注意**：必须先在 Kaggle Secrets 设置 `HF_TOKEN`

```bash
chmod +x export_to_hub.sh
./export_to_hub.sh
```

导出脚本会自动：
1. 转换 MaxText Orbax checkpoint 为 HuggingFace 格式（`safetensors`）
2. 上传到 `lyyh/llama3-200m-moe` 仓库
3. 模型可在 `https://huggingface.co/lyyh/llama3-200m-moe` 访问

### 6. 推理使用（可选）

```bash
# 下载模型
huggingface-cli download lyyh/llama3-200m-moe

# 使用 vLLM 推理
python -m "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained('lyyh/llama3-200m-moe', device='cuda')
tokenizer = AutoTokenizer.from_pretrained('lyyh/llama3-200m-moe')

prompt = 'Once upon a time, there was'
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
"
```

## 重要提示

### 显存限制（T4 15GB）

当前配置采用保守策略（`per_device_batch_size=4`），如果遇到 OOM：

1. **减小 batch size**：编辑 `configs/llama3_moe_200m_tinystories.yml`
   ```yaml
   per_device_batch_size: 2
   ```

2. **启用梯度累积**：增加 `gradient_accumulation_steps`
   ```yaml
   per_device_batch_size: 2
   gradient_accumulation_steps: 2
   ```

3. **减小序列长度**：`max_target_length: 1024`

### 数据集访问权限

- **TinyStories** 和 **Llama3 tokenizer** 都需要 HF 访问权限
- 确保在 Kaggle Secrets 设置了 `HF_TOKEN`
- 如果训练时遇到 401/403 错误，检查 token 是否有效

### W&B 日志

- W&B Entity：`gwhwh153td-china-times-org`
- 如果需要修改，编辑 `export_to_hub.sh` 和 `kaggle_train.sh` 中的 `entity` 和 `project`

### 模型参数估算

基于当前配置的参数量估算：
- Embedding 层：128k × 768 ≈ 98M
- 6 层 Transformer（主体）≈ 205M
- 总参数（含 embedding）：≈ 303M

这与 `base_emb_dim=768`, `vocab_size=128000` 的设计一致。

## 故障排除

### 训练启动失败

1. 检查日志文件 `training.log`
2. 确认依赖安装成功
3. 验证 `HF_TOKEN` 和 `WANDB_API_KEY` 已设置

### Checkpoint 转换失败

1. 检查 `maxtext_to_hf` 转换函数是否可用
2. 确认 checkpoint 路径正确

### HF Hub 上传失败

1. 验证 `HF_TOKEN` 权限足够（创建仓库和推送）
2. 检查网络连接（Kaggle 对 HF Hub 的访问）

## 技术细节

### MoE 层配置

MaxText 通过 `decoder_block: "mixtral"` 使用 Mixtral 风格的 MoE 层：
- 8 专家并行路由
- Top‑2 贪载均衡
- 支持 auxiliary loss（load balancing）

### 注意力类型

使用 MHA（Multi-Head Attention）而非 GQA（Grouped Query Attention）：
- `base_num_query_heads: 12`
- `base_num_kv_heads: 12`（相等，标准 MHA）

### 数据管道

MaxText 使用 HuggingFace 数据管道：
- Streaming 支持（避免下载全量数据）
- 自动 shuffle 和 tokenize
- 支持 `validation` split

### 编译优化

MaxText 使用 JAX/XLA 编译：
- 首次运行会有较长编译时间（10-30 分钟）
- 后续步骤会更快
- 如果想提前编译，运行 `train_compile.py`（可选）

## 参考资源

- [MaxText 文档](https://maxtext.readthedocs.io/en/latest/)
- [MaxText GitHub](https://github.com/AI-Hypercomputer/maxtext)
- [TinyStories 数据集](https://huggingface.co/datasets/roneneldanh/TinyStories)
- [HuggingFace Hub 指南](https://huggingface.co/docs/hub/)

## 许可证

本项目基于 Apache License 2.0 开源。

---

**生成时间**: 2025-02-05
**MaxText 版本**: 0.0.1（从源码克隆）

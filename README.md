## Implmentation steps
```bash
source osc_env.sh
```
### Step 1 先用 TokenGS DL3DV loader 取出一个 scene 的 images_all
We train exclusively on DL3DV (Ling et al. 2024) using 512 × 512 center-cropped
images.

```bash
self_implement_querysplat/TokenGS/scripts/inspect_dl3dv_scene.py
```

成功通过 TokenGS 的 Provider 读取一个真实 DL3DV scene，并取得：
    images_all.shape = [8, 3, 512, 512]
    dtype            = float32
    value range      = [0, 1]

```bash
cd /fs/scratch/PAS2099/Lemeng/NHT/self_implement_querysplat/TokenGS

conda run -n tokengs python scripts/inspect_dl3dv_scene.py \
  --data-root ../../DL3DV-10K-Sample \
  --scene-index 0 \
  --num-input-views 4 \
  --num-views 8 \
  --image-size 512 \
  --output-dir ../outputs/dl3dv_loader_smoke
```
### Step 2 固定 DL3DV sample 现在被明确拆成三个 QuerySplat 数据接口
images_all [8,3,512,512]
├── input_raw             [4,3,512,512]
├── input_normalized      [4,3,512,512]
└── supervision_images    [4,3,512,512]

```bash
self_implement_querysplat/TokenGS/tokengs/data/querysplat_images.py

input_raw = images_all[:num_input_views]
supervision_images = images_all[num_input_views:]
input_normalized = imagenet_normalize(input_raw)
```
input_raw 给 frozen VGGT，保持 [0,1]。
input_normalized 给 QuerySplat appearance RGB PatchEmbed。
supervision_images 作为 target-view rendering GT，保持 [0,1]。
逻辑独立于 DL3DV loader，将来 DataLoader 输出带 batch 维度时也能直接使用。
不使用 dataset camera，也不生成新的 Plücker；Plücker 后续必须来自 VGGT predicted cameras。

可以refer去 querysplat/scripts/infer.py
```bash
def build_model_input(images: torch.Tensor, decoder: ModelInputDecoder | None = None) -> ModelInput:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    normalized = normalize(images.clone()).unsqueeze(0)
    raw = images.unsqueeze(0)
    encoder = ModelInputEncoder(images_rgb=normalized, images_rgb_unnormalized=raw)
    if decoder is None:
        decoder = ModelInputDecoder(cam_view=torch.empty(0), intrinsics=torch.empty(0))
    return ModelInput(encoder=encoder, decoder=decoder)
```
### Step 3 接入官方 QuerySplat/VGGT，并建立 input-only VGM pass

直接复用官方 QuerySplat 的 `scripts/models`、`options.py`、`rendering`、`utils` 和 `third_party/vggt_omega`，避免重新实现导致 forward 逻辑、参数名或 checkpoint key 不一致。VGGT 权重放在 `TokenGS/checkpoints/vggt_omega_1b_512.pt`。

新增 `scripts/training/vggt_input_pass.py`，在不修改官方 `VGGTEncoder` 的前提下，用一次冻结的 VGGT Aggregator forward 同时提取 geometry features、input cameras、intrinsics、depth 和 confidence，避免训练时重复运行 1B encoder。单元测试已通过；真实 1B forward 需在 GPU 节点运行：

```bash
cd /fs/scratch/PAS2099/Lemeng/NHT/self_implement_querysplat/TokenGS
conda run -n querysplat python -m scripts.inspect_vggt_input_pass \
  --input-tensor ../outputs/dl3dv_loader_smoke/input_raw.pt \
  --config checkpoints/querysplat_vggto_1B_512_8192.yaml \
  --checkpoint checkpoints/vggt_omega_1b_512.pt \
  --output-dir ../outputs/vggt_input_smoke \
  --device cuda
```

### Step 4 Self-Calibrated Coordinate System - Sim(3)

对同一 scene 独立运行两次 frozen VGM：input-only pass 提供 `Fgeo` 并定义重建坐标系；all-view pass 只公开 cameras/intrinsics，不向重建分支传递 target-view features。使用两次 pass 中共同的 input cameras 估计 `all-view -> input-only` Sim(3)，再将 supervision cameras 对齐到 `Fgeo` 和 Gaussian 所在坐标系。

```bash
cd /fs/scratch/PAS2099/Lemeng/NHT/self_implement_querysplat/TokenGS
conda run -n querysplat python -m scripts.inspect_self_calibration \
  --images-all ../outputs/dl3dv_loader_smoke/images_all.pt \
  --num-input-views 4 \
  --config checkpoints/querysplat_vggto_1B_512_8192.yaml \
  --checkpoint checkpoints/vggt_omega_1b_512.pt \
  --output-dir ../outputs/self_calibration_smoke \
  --device cuda
```

### Step 5 Geometry/Appearance Dual-Branch Forward

新增 `scripts/training/dual_branch_forward.py`，在不修改官方 `querysplat.py` 的情况下复用 Step 4 的 `Fgeo` 和 input cameras，避免第三次 VGM forward。Geometry Queries 从 `Fgeo` 预测 center/scale/rotation；Appearance Queries 只读取 input RGB 与 input-camera Plücker features，预测 opacity/SH；最后使用 Sim(3)-aligned supervision cameras 渲染，保证 target images 不进入重建分支。

新增 `scripts/inspect_dual_branch_forward.py`：以 base-stage 的 1024 queries 生成 65,536 个 Gaussians，检查 BF16 forward、supervision rendering、L1 backward、三个可训练模块组的梯度以及 frozen VGGT 无梯度。相关单元测试、编译和 diff 检查已通过；真实 GPU smoke 不加载最终 QuerySplat checkpoint，因为此处验证的是随机初始化的 base-training graph。

```bash
cd /fs/scratch/PAS2099/Lemeng/NHT/self_implement_querysplat/TokenGS
conda run -n querysplat python -m scripts.inspect_dual_branch_forward \
  --images-all ../outputs/dl3dv_loader_smoke/images_all.pt \
  --input-normalized ../outputs/dl3dv_loader_smoke/input_normalized.pt \
  --num-input-views 4 \
  --config checkpoints/querysplat_vggto_1B_512_8192.yaml \
  --checkpoint checkpoints/vggt_omega_1b_512.pt \
  --output-dir ../outputs/dual_branch_smoke \
  --device cuda \
  --precision bf16
```

### Step 6 Complete Loss System

新增 `scripts/training/losses.py`，实现 `L1 + 0.2·LSSIM + λLPIPS(t)·LPIPS`、基于 input 与 aligned supervision cameras 的 visibility loss、input-only VGGT depth 反投影后的双向 Chamfer，以及 opacity-floor log-hinge。LPIPS 固定为 FP32；Sim(3)、visibility、Chamfer 和 opacity 数值路径也保持 FP32。LPIPS 渐入、Chamfer/opacity 渐出均由显式 linear schedule 控制；论文未给出的退火终点不写死在模型中。

Chamfer 对 depth points 和 Gaussian centers 做确定性采样并分块计算，避免构造完整的巨大距离矩阵。以下 smoke step 让 LPIPS、Chamfer 和 opacity 三个 scheduled weights 同时非零，只用于检查全部 loss 和 backward，不代表最终训练 schedule。

```bash
cd /fs/scratch/PAS2099/Lemeng/NHT/self_implement_querysplat/TokenGS
conda run -n querysplat python -m scripts.inspect_full_loss \
  --images-all ../outputs/dl3dv_loader_smoke/images_all.pt \
  --input-normalized ../outputs/dl3dv_loader_smoke/input_normalized.pt \
  --num-input-views 4 \
  --config checkpoints/querysplat_vggto_1B_512_8192.yaml \
  --checkpoint checkpoints/vggt_omega_1b_512.pt \
  --output-dir ../outputs/full_loss_smoke \
  --device cuda \
  --precision bf16 \
  --step 5000
```

### Step 7 Optimizer, LR, EMA, Checkpoint and Fixed-Scene Overfit

新增 `scripts/training/state.py`：沿用 TokenGS 的 AdamW 参数分组与 `(0.9,0.95)` betas，加入 linear warmup + cosine decay、global gradient clipping、仅覆盖 trainable parameters 的 EMA (`0.9995`) 和原子写入的可恢复 checkpoint。Checkpoint 保存 trainable model、optimizer、scheduler、EMA、global step 与 RNG；frozen VGGT 继续引用外部权重，不重复写入。

新增 `scripts/training/fixed_scene_cache.py` 与 `scripts/overfit_fixed_scene.py`。固定场景只运行一次 two-pass frozen VGM 并缓存中间层；每步仍重新执行 trainable layer fusion、双分支 decoder、完整 loss 和 backward。默认运行 100 steps，输出初始/最终 render、target、loss history 和可续训的单个 `latest.pt`。这个 cache 只用于 fixed-scene overfit 诊断，正式多场景训练不能跨 sample 复用。

```bash
cd /fs/scratch/PAS2099/Lemeng/NHT/self_implement_querysplat/TokenGS
conda run -n querysplat python -m scripts.overfit_fixed_scene \
  --images-all ../outputs/dl3dv_loader_smoke/images_all.pt \
  --input-normalized ../outputs/dl3dv_loader_smoke/input_normalized.pt \
  --num-input-views 4 \
  --config checkpoints/querysplat_vggto_1B_512_8192.yaml \
  --checkpoint checkpoints/vggt_omega_1b_512.pt \
  --workspace ../outputs/fixed_scene_overfit \
  --steps 100 \
  --warmup-steps 10 \
  --learning-rate 1e-4 \
  --gradient-clip 1.0 \
  --ema-decay 0.9995 \
  --precision bf16
```

Resume 必须保持总 steps 和 query 数一致：

```bash
conda run -n querysplat python -m scripts.overfit_fixed_scene \
  --workspace ../outputs/fixed_scene_overfit \
  --steps 100 \
  --warmup-steps 10 \
  --resume ../outputs/fixed_scene_overfit/latest.pt
```

`latest.pt` 包含约 10 亿 trainable parameters 的 model、AdamW states 和 EMA，文件可能达到十几 GB；脚本始终原子覆盖同一个文件，避免 checkpoint 数量累积。

### Step 8 Multi-Scene DL3DV Training with DDP

新增 `scripts/training/multiscene.py` 和 `scripts/train_querysplat.py`。训练直接读取 TokenGS `Provider(training=True)` 的 `images_all`；Provider 对每个 scene 和 epoch 重新采样帧，前 4 帧作为 input views，其余帧作为 supervision views。随后从同一份 `images_all` 重新构造 raw、ImageNet-normalized input 和 supervision tensors，防止不同分支采用不一致的裁剪或视图顺序。Step 8 固定为 base stage 的 4 input views、1024 queries；论文中的 2–12 views 和 query expansion 留到下一阶段。

每个 `torchrun` process 负责一张 GPU 和一个 sample，使用 `DistributedSampler` 划分 scenes。完整 two-pass VGM、双分支 decoder 和 renderer 都放在 DDP forward boundary 内；梯度同步后执行 global clipping、AdamW、warmup-cosine 和 EMA。loss scalars 在所有 ranks 求平均，只有 rank 0 写图像、JSONL 和原子 checkpoint。Checkpoint 额外保存每个 rank 的 RNG state，并要求恢复时 world size、view counts 和 dataloader 长度保持一致。

默认 base schedule 为 300K steps、2K warmup、30K 前退火 Chamfer/opacity，并在 10K–30K 渐入 LPIPS；论文没有给出后三个边界的精确数字，因此它们是显式 CLI 默认值，不是写死在模型里的论文常量。

同时将 `tokengs.utils` 的 evaluation metrics 改为 lazy import，并将 `Provider` 对 TokenGS Tyro options 的运行时导入改为 type-check-only。原因是 DL3DV loader 只需要 data utilities 和少量配置字段，不应被未使用的 `scikit-image` evaluation dependency 或旧 Tyro CLI 阻塞；调用 TokenGS evaluation API 时仍会按原接口加载 metrics。Loader resize 产生的 `1e-7` 量级 RGB 越界会在验证容差内 clamp 到 `[0,1]`，更大的数据范围错误仍直接报错。

先用单 GPU 做多场景 smoke（不是 fixed-scene cache，每一步都会为当前 scene 重新运行 two-pass VGM）：

```bash
cd /fs/scratch/PAS2099/Lemeng/NHT/self_implement_querysplat/TokenGS
conda run -n querysplat python -m scripts.train_querysplat \
  --data-root /fs/scratch/PAS2099/Lemeng/NHT/DL3DV-10K-Sample \
  --config checkpoints/querysplat_vggto_1B_512_8192.yaml \
  --vgm-checkpoint checkpoints/vggt_omega_1b_512.pt \
  --workspace ../outputs/querysplat_multiscene_smoke \
  --total-steps 20 \
  --warmup-steps 5 \
  --early-reg-end-step 10 \
  --lpips-start-step 5 \
  --lpips-ramp-end-step 10 \
  --checkpoint-every 20 \
  --image-every 10 \
  --num-workers 4 \
  --precision bf16
```

多 GPU 使用相同参数并由 `torchrun` 注入 rank 信息，例如 8 GPUs：

```bash
conda run -n querysplat torchrun --standalone --nproc-per-node=8 \
  -m scripts.train_querysplat \
  --data-root /path/to/full/DL3DV \
  --workspace ../outputs/querysplat_dl3dv_base \
  --total-steps 300000 \
  --num-workers 8 \
  --precision bf16
```

## Training Summary

当前 training code 实现的是 QuerySplat 的 **1024-query base stage**。它复用官方 QuerySplat inference model 和 TokenGS DL3DV loader，不改 checkpoint parameter names；VGGT-Ω backbone、camera head 和 depth head 始终冻结，QuerySplat geometry/appearance decoders、output heads、queries，以及 VGGT feature layer mixer/normalization参与训练。

### 1. Data and view sampling

DL3DV `Provider(training=True)` 每个 epoch 重新打乱 scenes，并为每个 scene 动态采样 8 张图。当前固定前 4 张为 input views、后 4 张为 supervision views；“动态”指每次采到的具体帧变化，而不是 input view 数变化。`images_all` 是唯一图像来源，并从中重新得到：

- `input_raw`：input-only VGM 使用的 `[0,1]` RGB。
- `input_normalized`：appearance RGB encoder 使用的 ImageNet-normalized input images。
- `supervision_images`：只用于 rendering loss 的 target images。

Loader resize 产生的 `1e-6` 内浮点越界会 clamp 到 `[0,1]`。代码要求 `images_all` 的前四张与 `input_raw` 逐元素相同，防止视图顺序或预处理不一致。

### 2. Two isolated VGM passes and Sim(3)

每个 multi-scene training step 都执行两次独立的 frozen VGGT-Ω forward：

1. **Input-only pass** 只读取 `input_raw`，提取第 4、11、17、23 层 geometry features，同时预测 input cameras、intrinsics 和 depth。它定义 geometry decoding 的原生坐标系。
2. **All-view pass** 读取 input 与 supervision views 的 union，但只导出 cameras/intrinsics，不向 reconstruction branch 传递 supervision features。

使用两个 pass 中共同的四个 input cameras 估计 `all-view → input-only` Sim(3)，再对齐全部 all-view cameras。最终 geometry features、input-camera Plücker rays、Gaussian centers 和 supervision cameras 位于同一坐标系，同时 target-view appearance/geometry information 不会泄漏到 reconstruction branches。

### 3. Geometry and appearance branches

Geometry branch 使用 1024 个 learnable geometry queries，通过 12 层 decoder cross-attend input-only geometry features。每个 query 输出 64 个 Gaussians 的 center、scale 和 rotation，因此 base stage 一共生成 65,536 个 Gaussians。

Appearance branch 从 `input_normalized` RGB patch embeddings 和 input VGM cameras 生成的 Plücker ray embeddings 构建 appearance features。它以 geometry tokens 为基础，再加入 learnable appearance queries，通过 6 层 decoder 输出 opacity 和一阶 spherical-harmonic color。监督视图只在 Gaussian 完成后作为 renderer cameras 和 target RGB 使用。

### 4. Rendering and losses

Gaussians 使用 Sim(3)-aligned supervision cameras 进行 differentiable rendering。总目标为：

```text
L = L1 + 0.2 * LSSIM + λLPIPS(t) * LLPIPS
    + 1.0 * Lvisibility
    + βCD(t) * LChamfer
    + βα(t) * Lopacity-floor
```

- `L1 + SSIM + LPIPS` 是主要 image-space reconstruction signal；LPIPS 始终用 FP32 计算。
- Visibility loss 使用 input 和 supervision cameras，惩罚位于所有 frusta 外或相机后方的 Gaussian centers。
- Chamfer 将 input-only VGGT depth 反投影为 pseudo point cloud，再与 Gaussian centers 计算双向距离；点云经过确定性采样和分块计算以控制显存。
- Opacity-floor 使用 log-hinge，早期阻止 Gaussians 过快透明化，默认 opacity floor 为 `0.1`。
- Chamfer 和 opacity-floor 只在训练早期使用并线性退火到零；LPIPS 延迟加入并线性升到 `0.05`。具体边界由 CLI 指定，不写死在模型中。

### 5. Optimization state

训练使用 AdamW，默认 learning rate `1e-4`、betas `(0.9, 0.95)`、weight decay `0.05`。bias、normalization 和一维参数不使用 weight decay。LR 先 linear warmup，再 cosine decay；global gradient norm clip 为 `1.0`。所有 trainable parameters 维护 decay `0.9995` 的 EMA，frozen VGGT 不进入 optimizer、EMA 或 training checkpoint。

Checkpoint 原子覆盖单个 `latest.pt`，包含 trainable model、AdamW、scheduler、EMA、global step、RNG 和训练 contract。Resume 要求 total steps、query/view counts、world size 和 dataloader steps-per-epoch 一致。由于 model、AdamW states 和 EMA 很大，checkpoint 约为 17 GB，写入时需要约两倍临时空间。

### 6. Fixed-scene and multi-scene modes

`scripts.overfit_fixed_scene` 是梯度与可学习性诊断：同一 scene 的 frozen VGM products 只计算并缓存一次，每步仍重新运行 trainable feature fusion、双分支 decoder、renderer 和 loss。它不能代表正式数据训练。

`scripts.train_querysplat` 是正式 multi-scene loop：每步读取新 sample，并为该 sample 重新执行完整 two-pass VGM 和 Sim(3)。当前每 GPU batch size 为 1。单 GPU直接运行时 `world_size=1`、`ddp=false`；使用 `torchrun` 时由 `DistributedSampler` 将 scenes 分到各 ranks，完整 forward 位于 DDP boundary 内，loss scalars 跨 ranks 求平均，只有 rank 0 写 metrics、images 和 checkpoint，各 rank 的 RNG state 都进入 checkpoint。

### 7. Current stage boundary

当前实现只覆盖论文的 base stage：4 input views、1024 queries、300K-step 接口。论文后续的 progressive query expansion（1024→2048→4096→8192）、每阶段 30K steps、随机 2–12 input views，以及 late-stage 95% loss-rank filtering 尚未进入当前 training loop，应作为下一阶段单独实现和验证。现有 11-scene `DL3DV-10K-Sample` 适合 smoke test 和小数据 overfit；训练可泛化模型需要完整 DL3DV 和更大的 global batch。

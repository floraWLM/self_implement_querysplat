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

```bash
conda run -n querysplat python -m scripts.overfit_fixed_scene \
  --images-all ../outputs/dl3dv_loader_smoke/images_all.pt \
  --input-normalized ../outputs/dl3dv_loader_smoke/input_normalized.pt \
  --num-input-views 4 \
  --config checkpoints/querysplat_vggto_1B_512_8192.yaml \
  --checkpoint checkpoints/vggt_omega_1b_512.pt \
  --workspace ../outputs/fixed_scene_overfit_500 \
  --steps 500 \
  --warmup-steps 20 \
  --early-reg-end-step 100 \
  --lpips-start-step 50 \
  --lpips-ramp-end-step 100 \
  --learning-rate 1e-4 \
  --gradient-clip 1.0 \
  --ema-decay 0.9995 \
  --precision bf16
```
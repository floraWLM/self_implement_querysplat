## Implmentation steps
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




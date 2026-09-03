### Implmentation steps
## Step 1 先用 TokenGS DL3DV loader 取出一个 scene 的 images_all
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
## Step 2 固定 DL3DV sample 现在被明确拆成三个 QuerySplat 数据接口
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

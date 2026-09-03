### Implmentation steps
## Step 1 先用 TokenGS DL3DV loader 取出一个 scene 的 images_all
self_implement_querysplat/TokenGS/scripts/inspect_dl3dv_scene.py
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

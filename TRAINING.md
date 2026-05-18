# MapDet3D-Mono-DepthPC-Last — Training Guide

**How to reproduce, fine-tune, or extend the model.**

| | |
|---|---|
| Document version | 2.0 (self-contained) |
| Companion docs | [`../README.md`](../README.md), [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Audience | Engineers training or fine-tuning MapDet3D-Mono-DepthPC-Last |
| Prerequisites | A working PyTorch ≥ 2.2 environment, ≥ 1× GPU with ≥ 24 GB VRAM, ~360 GB of disk for cached features + datasets, and basic familiarity with multi-dataset training and pinhole-camera intrinsics. |

---

## Table of contents

1. [Overview](#1-overview)
2. [Hardware and compute](#2-hardware-and-compute)
3. [Environment setup](#3-environment-setup)
4. [Datasets — depth and intrinsics requirements](#4-datasets--depth-and-intrinsics-requirements)
5. [The 30-class taxonomy](#5-the-30-class-taxonomy)
6. [Preprocessing](#6-preprocessing)
7. [The 3-stage joint curriculum](#7-the-3-stage-joint-curriculum)
8. [Hyperparameters](#8-hyperparameters)
9. [Feature caching](#9-feature-caching)
10. [Evaluation protocols](#10-evaluation-protocols)
11. [Fine-tuning on a new dataset](#11-fine-tuning-on-a-new-dataset)
12. [Checkpoints](#12-checkpoints)
13. [Troubleshooting](#13-troubleshooting)
14. [Reproducibility checklist](#14-reproducibility-checklist)

---

## 1. Overview

You will:

1. Download Hypersim + WildDet3D-indoor + ARKit val.
2. Preprocess: 9-DoF → 7-DoF (Hypersim), 12-dim decode (WildDet3D), 30-class filter, canonical normalization. **Plus**, for this variant: convert Hypersim depth from ray-length to optical-axis convention (§6.2b) and filter WildDet3D-indoor to images with sensor depth available (§6.3b).
3. (Recommended) Pre-cache MapAnything's single-view encoder features.
4. Train in three stages: (S1) Hypersim warm-up, (S2) joint training, (S3) real-data emphasis.
5. Evaluate on Hypersim val, WildDet3D-indoor-with-depth val, and zero-shot on ARKitScenes val.

**Wall-clock estimate at 8 × H100 80 GB:** ~45 hours total (S1 ≈ 10 h, S2 ≈ 28 h, S3 ≈ 7 h). This is the fastest of the MapDet3D family because both architectural simplifications compound: smaller cross-attention activations (single-layer memory) and smaller effective dataset (depth-availability filter).

---

## 2. Hardware and compute

| Configuration | What you can do |
|---|---|
| 1 × RTX 3090 (24 GB) | Inference only |
| 1 × A100 40 GB | Fine-tune 3D head on small data; comfortable thanks to lower memory footprint |
| 4 × A100 40 GB | Full training; ~4 days |
| 8 × H100 80 GB | Full training; ~2 days. **Recommended for reproduction.** |

The frozen MapAnything backbone is the memory hog: ViT-G + 16-layer AAT runs at ~22 GB per forward in bf16 on a 518² image. The detection heads add ~5 GB. Single-layer memory cuts ~17 GB of decoder activations relative to a multi-scale baseline. Per-GPU batch of 8 single-view samples fits comfortably on 80 GB; on 40 GB cards, per-GPU batch 4 works.

---

## 3. Environment setup

```bash
conda create -n mapdet3d-mono-depthpc-last python=3.11 -y
conda activate mapdet3d-mono-depthpc-last

# PyTorch (matched to your CUDA)
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121

# MapAnything (frozen backbone)
pip install -e git+https://github.com/facebookresearch/map-anything.git#egg=mapanything

# MGIoU loss
pip install mgiou

# Other dependencies
pip install \
    timm scipy einops h5py pillow numpy tqdm omegaconf hydra-core \
    open3d opencv-python matplotlib

# (Optional) Hypersim preprocessing toolkit
pip install hypersim-tools
```

Test:
```bash
python -c "from mapanything.models.mapanything import MapAnything; print('ok')"
python -c "from mgiou import MGIoU3D; print('ok')"
```

---

## 4. Datasets — depth and intrinsics requirements

Because this variant unprojects depth at every training step, **every training sample needs both `depth` and `K`**. Below is how that requirement plays out per dataset.

### 4.1 Hypersim — primary synthetic source

| | |
|---|---|
| Source | [`apple/ml-hypersim`](https://github.com/apple/ml-hypersim) |
| Training images (indoor, post-filter) | ~46.6 k |
| Depth source | **GT rendered depth** (`frame.depth_meters.hdf5`), metric, dense, **no holes** |
| Intrinsics | Per-scene `_detail/cam_intrinsics.csv`, deterministic |
| Train-time `valid_mask` | All-ones (synthetic depth is dense) |
| Disk | ~150 GB |

Hypersim is *ideal* for this variant: rendered depth is perfectly dense and metric; intrinsics are precisely known. No holes during S1; the depth-noise + depth-hole augmentations (§8.4) inject sensor-realism during S2/S3.

> **Critical gotcha:** Hypersim's `depth_meters.hdf5` stores depth as **distance along the ray** (= ray length), not along the optical axis. Our model expects depth-along-axis. The conversion runs once at preprocessing (§6.2b). Skipping this step quietly makes every Hypersim point cloud wrong by 1–5%, with the largest bias at image corners.

### 4.2 WildDet3D-indoor — primary real-world source (with depth-availability filter)

| | |
|---|---|
| Source | [`allenai/WildDet3D`](https://github.com/allenai/WildDet3D) |
| Training images (indoor, with sensor depth) | ~33–37 k (after the depth-availability filter described below) |
| Depth source | Inherited from the ingested source datasets (SUN-RGBD, ScanNet, ARKit, Objectron, CA-1M); in-the-wild human-verified images are dropped |
| Intrinsics | Per-image in WildDet3D metadata |

**The central decision for this variant.** WildDet3D-Data was assembled from multiple sources. Some carry sensor depth (Kinect, Structure, ARKit LiDAR, iPhone LiDAR); some in-the-wild human-verified images carry no sensor depth.

We drop the no-depth subset entirely. The rationale: at deployment time the input has sensor depth — if we trained with predicted depth in 30%+ of samples, the model would learn biases that don't appear at inference. Keeping train and deployment distributions matched is worth the data loss.

| Per-image source tag | Train-time depth available? | Action |
|---|---|---|
| `sun_rgbd` (Kinect) | yes | KEEP |
| `scannet` (Structure) | yes | KEEP |
| `arkit` (LiDAR) | yes | KEEP at the source-tag level (ARKit val frames are physically held out for eval) |
| `objectron` (ARKit) | yes | KEEP |
| `ca-1m` (iPhone LiDAR) | yes | KEEP |
| `hypersim` | yes (already in Hypersim split) | KEEP |
| `wild_human_verified` (in-the-wild) | no | **DROP** |
| `synthetic` (SAM3+FoundationPose) | — | DROP (we use human-verified only anyway) |

After applying the filter, the effective WildDet3D-indoor training set is ~33–37 k images. The eval split is unaffected (WildDet3D-Bench val already requires depth).

### 4.3 ARKitScenes — zero-shot eval only

| | |
|---|---|
| Source | [`apple/ARKitScenes`](https://github.com/apple/ARKitScenes) |
| Val frames | ~5.6 k |
| Depth source | iPad Pro LiDAR (sparse but accurate; ~10–15% missing pixels on textureless / glossy surfaces) |
| Intrinsics | Per-frame, provided in the official release |

ARKit is the canonical deployment scenario for this variant. LiDAR depth is sparse-but-clean; the snap-to-valid fallback in the query generator handles the missing pixels.

### 4.4 Train-time data summary

| Dataset | Effective train size | Notes |
|---|---|---|
| Hypersim | ~46.6 k | Full public release; convert depth ray → axis |
| WildDet3D-indoor (filtered) | ~33–37 k | Depth-availability filter applied |
| **Total** | **~80–84 k** | |

---

## 5. The 30-class taxonomy

We train and evaluate on a fixed set of **`C = 30`** indoor classes. The architecture is class-list-agnostic — changing `C` is a config edit that resizes the final classifier head, nothing else in the pipeline depends on specific class identities.

**Recommended construction:**
```
30 classes = ARKit-17 furniture categories
           ∪ 13 most-frequent NYU40 indoor classes not in ARKit-17
```

**Filtering rule.** Annotations whose source-dataset class does not map to any of the 30 are **dropped entirely** at preprocessing time. Symmetric across both datasets. No "other" / catch-all bucket.

**Class-list file.** Store the 30 names in `data/classes.txt`, one per line; first line is `background` (class id 0). Mapping files:
```json
// data/hypersim_to_30class.json
{ "5": 1, "7": 2, ..., "9": null }   // null = drop
```
```json
// data/wilddet3d_to_30class.json
{ "chair": 1, "office chair": 1, ..., "stool": null }
```

**Generating the mappings:**
```bash
python scripts/build_30class_mappings.py \
    --classes-file data/classes.txt \
    --hypersim-meta data/hypersim/_detail/ \
    --wilddet3d-meta data/wilddet3d/metadata/ \
    --output-dir data/
```
Prints per-class counts; sanity-check that no class has < 100 annotations across the combined training set.

**Changing `C`.** Only three things change: `data/classes.txt`, the two mapping JSONs, the `num_classes` field in the config YAML.

---

## 6. Preprocessing

Run these once per dataset.

### 6.1 Hypersim — 9-DoF → 7-DoF box projection

Hypersim's 3×3 rotation may include shear or non-uniform scale that doesn't reduce to an OBB cleanly. Use SVD to project to the closest 7-DoF OBB:
```python
def project_9dof_to_7dof(R_3x3, scale_3, position_3):
    M = R_3x3 @ np.diag(scale_3)
    U, S, Vt = np.linalg.svd(M)
    R_orth = U @ Vt
    size   = S
    residual = np.linalg.norm(M - R_orth @ np.diag(S)) / (np.linalg.norm(M) + 1e-8)
    if residual > 0.05:
        return None  # drop this annotation
    return R_orth, size, position_3
```
Drops ~5–10% of Hypersim's 3D boxes (heavily-sheared ones — shelves, curtains).

### 6.2 Hypersim — 2D boxes from instance masks

Connected-components → bounding-rectangle on each instance mask:
```python
def extract_2d_box(instance_mask, instance_id):
    ys, xs = np.where(instance_mask == instance_id)
    if len(xs) < 16:
        return None
    return [xs.min(), ys.min(), xs.max(), ys.max()]
```

### 6.2b ⚠️ Hypersim depth: ray-length → optical-axis conversion (DepthPC-specific, mandatory)

```bash
python scripts/hypersim_depth_to_axis.py \
    --input data/hypersim/scenes/ \
    --intrinsics data/hypersim/_detail/cam_intrinsics.csv \
    --output data/hypersim/depth_axis/
```

The script per-pixel divides by `sqrt(1 + ((u-cx)/fx)² + ((v-cy)/fy)²)` to convert from ray-length to z-along-optical-axis. Output is a parallel set of HDF5 files: `depth_axis.hdf5` alongside the original `depth_meters.hdf5`.

> Without this step every unprojected 3D point is biased by 1–5% radially outward, with the largest error at the image corners. Training looks superficially OK (loss decreases) but the 3D head learns to compensate, baking in dataset-specific bias that doesn't transfer to real sensors.

### 6.2c Hypersim — map semantic labels to 30 classes

Apply `data/hypersim_to_30class.json`. Drop unmatched instances.

### 6.3 WildDet3D — decode 12-dim boxes, map to 30 classes

Each 3D box is a 12-dim vector `(Δcx, Δcy, log d̂, log ŵ, log ĥ, log l̂, r1..r6)` where center is normalized by sc=10 and dimensions by sdim=2.0:
```python
def decode_12dim(box_12, intrinsics):
    dcx, dcy, log_d, lw, lh, ll, *rot6 = box_12
    d = np.exp(log_d / 2.0)
    w, h, l = np.exp(np.array([lw, lh, ll]) / 2.0)
    cx, cy = decode_image_center(dcx, dcy, intrinsics)
    center_3d = unproject(cx, cy, d, intrinsics)
    R = gram_schmidt(rot6)
    return center_3d, (w, h, l), R
```
Apply `data/wilddet3d_to_30class.json`. Strings mapping to `null` (or not present) are dropped.

### 6.3b ⚠️ WildDet3D — depth-availability filter (DepthPC-specific, mandatory)

```bash
python scripts/wilddet3d_depth_filter.py \
    --metadata data/wilddet3d/metadata/ \
    --output data/wilddet3d/indoor_with_depth_manifest.parquet \
    --require-sensor-depth
```

Reads WildDet3D's source-dataset metadata; keeps only images tagged from `sun_rgbd`, `scannet`, `arkit`, `objectron`, `ca-1m`, `hypersim`. Drops `wild_human_verified` (no sensor depth) and `synthetic`. Expected output: ~33–37 k images with depth + K.

Log per-source counts; sanity-check that no single source dominates (> 50%) — if one does, the model will overfit to that source's depth-noise distribution.

### 6.4 Canonical rotation normalization

Enforce `w ≤ l` and yaw ∈ `[0, π)` on all GT 3D boxes:
```python
def canonicalize(center, size, R):
    w, l, h = size
    if w > l:
        w, l = l, w
        R = R @ rotation_z(np.pi / 2)
    yaw = extract_yaw(R)
    if yaw >= np.pi:
        R = R @ rotation_z(-np.pi)
    return center, (w, l, h), R
```
Eliminates the 4-fold yaw symmetry that would otherwise make the rotation loss bimodal.

### 6.5 K-rescale-for-letterbox utility (DepthPC-specific)

Used at both train and eval time:
```python
def rescale_K_for_letterbox(K_orig, H_orig, W_orig, target=518):
    """Compute the K that matches the letterbox-resized image."""
    scale = target / max(H_orig, W_orig)
    pad_h = (target - H_orig * scale) / 2
    pad_w = (target - W_orig * scale) / 2
    K = K_orig.copy().astype(np.float32)
    K[0, 0] *= scale
    K[1, 1] *= scale
    K[0, 2] = K_orig[0, 2] * scale + pad_w
    K[1, 2] = K_orig[1, 2] * scale + pad_h
    return K
```
Lives in `mapdet3d_mono_depthpc_last.preprocess`.

---

## 7. The 3-stage joint curriculum

Training proceeds in three stages, progressively introducing real-world data on top of a clean synthetic warm-up:

| Stage | Data mix | Epochs | LR | Warmup | Grad clip | MGIoU λ |
|---|---|---|---|---|---|---|
| **S1: Hypersim warm-up** | 100% Hypersim | 8 | 1e-4 | 1k steps | 35.0 | 0 → 2 over first 7k steps |
| **S2: Joint** | 50/50 Hypersim / WildDet3D-indoor-with-depth | 16 | 1e-4 | 500 steps | 35.0 | 2 |
| **S3: Real emphasis** | 25/75 Hypersim / WildDet3D-indoor-with-depth | 6 | 1e-5 | none | 0.1 (stability fix) | 2 |

**Total: 30 epochs.**

**Sampling.** Per training step, sample one scene from each dataset proportional to the stage mix, then sample one image. Use a `DatasetMixer` that interleaves at the batch level (each batch is from one dataset) to avoid mixing label-space tensors in a single Hungarian matching call.

**Teacher-forcing schedule** for the 2D head → query generator handoff: 100% GT 2D boxes in S1, anneal 100% → 0% over S2, 0% in S3.

**Why this curriculum.**
- S1 establishes a clean loss landscape on dense synthetic data. MGIoU is L1-warmed; rotation head sees clean GT.
- S2 introduces real-world data while the model is partially trained; domain shift absorbed gradually.
- S3 emphasizes real data because deployment is on real cameras. The LR drop + small grad clip is a stability fix.

**DepthPC-specific monitor.** Log average `K_eff / K` (fraction of query slots successfully filled — see ARCHITECTURE.md §4.5) per epoch. Some images will pass below the warning threshold (0.7) because of depth sparsity — this is not a training error. If the average across a full S2 epoch drops below 0.6, the depth-availability filter wasn't strict enough; fix the filter rather than the training.

---

## 8. Hyperparameters

### 8.1 Optimizer and schedule

| Setting | Value | Notes |
|---|---|---|
| Optimizer | AdamW | β = (0.9, 0.999) |
| Weight decay | 1e-4 | |
| Base LR | 1e-4 (S1/S2), 1e-5 (S3) | |
| Scheduler | cosine within each stage, η_min = 1e-6 | |
| Precision | bfloat16 | |
| LayerNorm precision | fp32 (promoted) | Required for bf16 numerical stability |
| Per-GPU batch | 8 single-view samples (H100), 4 (40 GB) | |
| Gradient clip | 35.0 in S1/S2, 0.1 in S3 | |
| Input resolution | 518 × 518 | DINOv2 patch=14; 37×37 token grid |

### 8.2 Model

| Setting | Value |
|---|---|
| Backbone | MapAnything frozen, **features-only** |
| Geometric heads | Skipped at inference (`backbone_geometry_heads=False`) |
| Number of views N | 1 |
| pts3d source | **Unprojection of depth + K** (no learned pts3d consumed) |
| 2D head | 4 layers, 256-dim, 100 queries |
| 3D head | 8 layers, 512-dim, FFN 512, K=256 queries |
| Box parameterization | 12 dims (Δcenter, log_size, rot6d) |
| Memory tokens | **1369** (aat_feat[15] only) |
| Memory PE | PE3D from depth-unprojected pts3d; `missing_pe` for invalid tokens |

### 8.3 Loss weights

| Component | Weight |
|---|---|
| 2D classification (focal) | λ_2D_cls = 2 |
| 2D L1 | λ_2D_l1 = 5 |
| 2D GIoU | λ_2D_giou = 2 |
| 3D classification (focal) | λ_cls = 2 |
| 3D center L1 | λ_ctr = 5 |
| 3D log_size L1 | λ_sz = 1 |
| 3D rotation chord | λ_rot = 2 |
| 3D MGIoU | λ_box = 0 → 2 (warmup) |
| 3D aleatoric NLL | λ_unc = 0.1 (V3 only) |
| 3D denoising | λ_dn = 1 (V3 only) |
| `missing_pe` L2 prior | λ_missing_pe_l2 = **0.05** (5× tighter than would be reasonable for a multi-scale memory; see ARCHITECTURE.md §6.6) |

Per-class focal-loss α: uniform 0.25.

### 8.4 Augmentations

| Type | Setting |
|---|---|
| Color jitter | brightness 0.2, contrast 0.2, saturation 0.2 |
| Gamma | uniform(0.7, 1.3) |
| Horizontal flip | p=0.5 (mirror image and 3D boxes about camera x-axis) |
| No vertical flip | gravity matters indoors |
| Per-dataset color matching in S2 | match WildDet3D channel mean/std to Hypersim's |
| **Depth noise injection (DepthPC-specific)** | σ = 1% of depth value, additive Gaussian, S2/S3 only. Augments Hypersim's clean depth with sensor-like noise. Disabled in S1 (warmup uses clean depth). |
| **Random depth-hole injection (DepthPC-specific)** | drop random 5% of pixels (set to 0) in Hypersim during S2/S3. Forces the snap-to-valid fallback to be exercised during training so the model is robust to it at inference. |

### 8.5 Combined config file

```yaml
# configs/mono_depthpc_last_v1.yaml (key fields)

data:
  hypersim:
    depth_source: "depth_axis"            # use the optical-axis-converted depth
  wilddet3d:
    require_sensor_depth: true            # depth-availability filter

model:
  pts3d_source: "unproject_depth"         # not "mapanything_pts3d"
  intrinsics_required: true
  backbone_geometry_heads: false          # skip at inference

  memory:
    sources: ["aat_15"]                   # single-layer memory
    use_level_id_embedding: false
  query_generator:
    roi_features: "aat_15"
    snap_to_valid: true
    drop_query_if_no_valid_pixel: true

training:
  augmentation:
    depth_noise_sigma_rel: 0.01
    depth_hole_drop_prob: 0.05
  memory_pe:
    use_missing_pe: true
    missing_pe_l2_lambda: 0.05            # tighter than the 0.01 default
```

---

## 9. Feature caching

The frozen backbone dominates wall-clock time. Pre-compute single-view DINOv2 encoder features to disk:

```bash
python scripts/cache_features.py \
    --dataset hypersim \
    --output-dir cache/hypersim/ \
    --batch 16 \
    --gpus 0,1,2,3,4,5,6,7
```

**What's cached.** Only the encoder features `enc_feat` (37, 37, 1536) per image, bf16. **NOT** the AAT features — they depend on the depth input (consumed by the AAT as a prior) and would have to be recomputed when depth is augmented.

| Dataset | Cache size |
|---|---|
| Hypersim (46.6 k images) | ~190 GB |
| WildDet3D-indoor-with-depth (~35 k images) | ~150 GB |
| ARKit val (5.6 k images) | ~23 GB |
| **Total** | **~360 GB** |

**No need to cache `pts3d`.** It's a deterministic function of (depth, K) — recompute at dataloader time. Cost is negligible CPU.

Use local NVMe; spinning disk will bottleneck training.

---

## 10. Evaluation protocols

### 10.1 Hypersim val (in-domain)
- mAP@0.25 and mAP@0.50 on 3D IoU.
- Per-class AP for all 30 named classes.

### 10.2 WildDet3D-indoor-with-depth val (in-domain)
- Center-distance matching at 0.5 m / 1.0 m thresholds (WildDet3D's convention).
- Closed-vocab AP over the 30 named classes. Val annotations outside the 30 are excluded from eval.

### 10.3 ARKitScenes val (zero-shot — the headline result)
- Standard ARKit split (5.6 k val frames).
- mAP@0.25 and mAP@0.50 on 3D IoU.
- 1:1 class mapping (ARKit's 17 ⊆ our 30).

### 10.4 DepthPC-specific monitor
Track average `K_eff / K` across the eval set. If it drops below 0.7 on any eval split, the snap-to-valid fallback is being heavily exercised — inspect the depth holes' spatial distribution before drawing conclusions about model quality.

### 10.5 Per-class small-object diagnostic
Track per-class AP for `{lamp, monitor, pillow, tv}` separately. If small-object AP drops disproportionately compared to large-object AP, the multi-scale fusion in a baseline was providing real localization signal — see the 2-level fallback noted in Troubleshooting (§13).

### 10.6 Reporting target
The published bar for zero-shot 3D detection: ≥ 50% of an in-domain-trained model's ARKit AP.

---

## 11. Fine-tuning on a new dataset

For a new RGB-D deployment (warehouse, hospital, vendor-specific furniture set). **Every sample must have depth + K** — this variant doesn't support intrinsics-free inputs. If your new dataset lacks intrinsics, use a sibling variant that has a learned pts3d head.

Steps:
1. **Map labels.** Either map your classes to a subset of the 30 (reuse the published checkpoint), or define a new `data/classes.txt` and re-train (you lose the published classifier head but keep backbone + decoder).
2. **Convert annotations.** Output 7-DoF OBBs + 2D boxes per image; apply canonical normalization (§6.4).
3. **Verify depth + K availability per sample.** Drop samples lacking either.
4. **Configure.** Copy `configs/finetune_template.yaml`; point `data_root` at your dataset; set `init_checkpoint=checkpoints/mapdet3d-mono-depthpc-last-v1.pth`.
5. **Train.** Run a shortened curriculum: skip S1, run S2 for 4 epochs (50/50 your data + WildDet3D-indoor-with-depth for regularization), then S3 for 2 epochs (100% your data).
   ```bash
   python train.py --config configs/finetune_yourdataset.yaml \
                   --resume checkpoints/mapdet3d-mono-depthpc-last-v1.pth
   ```

Wall-clock estimate (8 × H100, 5 k images): ~5 hours.

**Don't** fine-tune the MapAnything backbone unless you have a lot of new data (≥ 100 k images).

---

## 12. Checkpoints

| Name | Size | Notes |
|---|---|---|
| `mapdet3d-mono-depthpc-last-v1.pth` | ~250 MB (heads only) | Trained jointly on Hypersim + WildDet3D-indoor-with-depth |

The MapAnything backbone is NOT included (~7 GB, licensed separately). At load time the framework downloads MapAnything's Apache-2.0 weights from the official HF hub and combines.

Loading:
```python
from mapdet3d_mono_depthpc_last import MapDet3DMonoDepthPCLast
model = MapDet3DMonoDepthPCLast.from_pretrained("mapdet3d-mono-depthpc-last-v1")  # auto-downloads backbone
```

---

## 13. Troubleshooting

### Training diverges in S2 (loss → NaN)
- Verify MGIoU warmup is enabled (default).
- Verify bf16 LayerNorm precision is fp32-promoted.
- DepthPC-specific: check that `K_eff / K` isn't 0 for some batches (would mean all queries got dropped → empty Hungarian matching). If yes, investigate the dataset for corrupt depth files or wrong K.

### `||missing_pe||` stays much smaller than other memory PE norms by mid-S2
- The `missing_pe` vector is being over-regularized in this variant (smaller memory + only a fraction of tokens invalid = ~4× less gradient signal than in a multi-scale baseline).
- Mitigation: walk the L2 prior back from 0.05 to 0.01.

### Small-object AP is much worse than expected
- Predicted failure mode of the single-layer memory: the encoder features in a multi-scale baseline were contributing sharp localization for small objects.
- Mitigation: try the 2-level fusion fallback (`enc_feat + aat_feat[15]`) — a config-only change:
  ```yaml
  model:
    memory:
      sources: ["enc", "aat_15"]
  ```
  Re-train S3 with the broader memory; expect a small AP recovery on small objects.

### Hypersim 3D box loss is much higher than WildDet3D's
- Expected for the first few hundred steps of S2. Hypersim has more boxes per image (~10) than WildDet3D (~3); per-image loss is naturally larger.
- If it doesn't equalize after 1 epoch of S2, reduce the Hypersim share in S2 to 40%.

### 2D head's gradient dominates 3D head's
- Visible as `||∇_2D|| / ||∇_3D|| > 2.0` for >100 steps.
- GradNorm auto-rebalancing should kick in. If not, set `λ_2D_*` to 0.5× defaults manually.

### Zero-shot ARKit mAP is much lower than expected
- First check the class mapping is correct.
- Second check: depth input correctly aligned and in meters.
- Third check: the 2D head's recall on ARKit. If < 60%, lower `score_threshold` at inference, or fine-tune the 2D head on a few hundred ARKit images.

### Hypersim depth and unprojected pts3d give visually wrong 3D shapes
- Did you run `scripts/hypersim_depth_to_axis.py` (§6.2b)? If you skipped this, pts3d is biased radially. Run it.

### WildDet3D images keep producing `K_eff = 0`
- Most likely cause: intrinsics stored in a different convention than OpenCV. WildDet3D's metadata sometimes uses normalized intrinsics (cx, cy in [0,1]). If K[0,2] < 5 in your data, that's normalized — multiply by image width to get pixels.

### Caching uses too much disk
- AAT taps are already excluded from the cache.
- Reduce input resolution from 518 to 392 (28 × 14): cache shrinks ~50%; mAP drops ~1–2 points.

### My new dataset doesn't have intrinsics, only depth
- This variant cannot be used. Use a sibling variant that has a learned pts3d head (MapAnything's predicted pts3d does not require user-provided K).

### Some of the 30 classes have very few training annotations
- Sanity-check after preprocessing: `python scripts/build_30class_mappings.py` prints per-class counts. If any class has < 100, drop it from `data/classes.txt` or augment the WildDet3D synonym list.

### Too many annotations get dropped — concerned about data efficiency
- Target ≥ 60 k 3D boxes remaining across Hypersim + WildDet3D-indoor-with-depth combined. If you have far fewer, verify the WildDet3D mapping covers common synonyms and the Hypersim NYU40 IDs are mapped correctly.

---

## 14. Reproducibility checklist

- [ ] Set `torch.use_deterministic_algorithms(True)` in `train.py`.
- [ ] Seed all RNGs: torch, numpy, python random, CUDA.
- [ ] Pin the MapAnything checkpoint version (commit hash in `config.yaml`).
- [ ] Pin the `mapdet3d_mono_depthpc_last` git commit (`output_dir/git_sha.txt`).
- [ ] Log every hyperparameter to `output_dir/config.yaml` (OmegaConf).
- [ ] Save `data/classes.txt` + both mapping files + the depth-filter manifest (`indoor_with_depth_manifest.parquet`) to `output_dir/`.
- [ ] Log per-class annotation counts pre- and post-depth-filter.
- [ ] Log average `K_eff / K` per epoch to TensorBoard.
- [ ] Log `||missing_pe||` per epoch (early-warning for over-regularization).
- [ ] After training, run eval on Hypersim val, WildDet3D-indoor-with-depth val, and ARKit val (zero-shot).

---

*End of training guide.*

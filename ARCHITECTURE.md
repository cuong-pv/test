# MapDet3D-Mono-DepthPC-Last — Architecture Specification

**Single-image RGB-D 3D object detection. Depth-derived point cloud + last-AAT-layer features.**

| | |
|---|---|
| Document version | 2.0 (self-contained) |
| Companion docs | [`../README.md`](../README.md), [`TRAINING.md`](TRAINING.md) |
| Audience | ML engineers implementing, modifying, or reviewing the model |
| Prerequisites | Familiarity with DETR, transformer encoders, basic 3D box geometry, pinhole camera model. Knowledge of MapAnything is *not* required — relevant facts are included inline. |

---

## Table of contents

1. [Problem and scope](#1-problem-and-scope)
2. [Input / output contract](#2-input--output-contract)
3. [System diagram](#3-system-diagram)
4. [Module spec](#4-module-spec)
   - 4.1 [MapAnything backbone (features-only, N=1)](#41-mapanything-backbone-features-only-n1)
   - 4.2 [Depth-to-point-cloud unprojection](#42-depth-to-point-cloud-unprojection)
   - 4.3 [Single-layer feature memory](#43-single-layer-feature-memory)
   - 4.4 [2D detection head](#44-2d-detection-head)
   - 4.5 [Query generator (depth-PC anchored, snap-to-valid)](#45-query-generator-depth-pc-anchored-snap-to-valid)
   - 4.6 [3D decoder](#46-3d-decoder)
   - 4.7 [Box parameterization and prediction heads](#47-box-parameterization-and-prediction-heads)
   - 4.8 [Losses and matching](#48-losses-and-matching)
5. [Coordinate conventions and unit handling](#5-coordinate-conventions-and-unit-handling)
6. [Why these choices (intuition + evidence)](#6-why-these-choices-intuition--evidence)
7. [Tensor shape reference](#7-tensor-shape-reference)
8. [Risks, failure modes, and mitigations](#8-risks-failure-modes-and-mitigations)
9. [Pseudocode (single inference step)](#9-pseudocode-single-inference-step)
10. [References](#10-references)
11. [Glossary](#11-glossary)

---

## 1. Problem and scope

**Task.** Given a single RGB image, an aligned metric depth map, and 3×3 camera intrinsics `K`, predict the set of 3D oriented bounding boxes for all objects belonging to a fixed closed-vocab set of **`C = 30`** indoor classes.

**Two key architectural commitments.**

1. The 3D point cloud used for query anchoring and positional encoding is computed by **unprojecting the input depth map with `K`** — a deterministic geometric operation, not a network output. MapAnything is used as a visual feature backbone, but its geometric heads (`pts3d`, `depth_along_ray`, `cam_*`, `metric_scaling_factor`) are not consumed downstream.
2. The 3D decoder's cross-attention memory is built from **a single feature map** — MapAnything's final AAT layer (layer 15) at 37×37 resolution — not a fusion of multiple feature taps.

**Why this matters.**
- Metric scale is set deterministically by the sensor. With a calibrated sensor, this removes a class of failure modes that affect models that learn to output `pts3d`.
- The decoder learns "given a point in 3D, what object is around it?" without also having to infer "what is the metric scale of this scene?"
- The single-layer memory is ~4× smaller than a multi-scale memory, cutting cross-attention compute and training memory accordingly.

**Inputs at inference time.**
- One RGB image (any aspect ratio; resized internally to 518 × 518).
- One depth map aligned to the RGB image, **metric units (meters)**, missing pixels signalled by `0.0` or `NaN`.
- **`K` ∈ ℝ³ˣ³** — required. No fallback to a learned predictor in this variant.

**Outputs.** A list of `(center_m, size_m, R, class, score)` boxes in the camera frame — schema in [§2.4](#24-output-schema).

**Out of scope.** Open-vocab; outdoor / LiDAR; real-time inference; multi-view fusion; RGB-only operation (no depth or no intrinsics → use the `MapDet3D-Mono` variant).

---

## 2. Input / output contract

### 2.1 RGB image

| Property | Value |
|---|---|
| Shape | `(H, W, 3)` or PIL; converted internally to `(1, 3, 518, 518)` |
| Dtype | `uint8` accepted; converted to `float32` |
| Color order | RGB |
| Resize | Bicubic to 518 × 518 with letterbox padding |
| Normalization | DINOv2 standard: `mean = (0.485, 0.456, 0.406)`, `std = (0.229, 0.224, 0.225)` |

### 2.2 Depth map

| Property | Value |
|---|---|
| Shape | `(H, W)` matching RGB; **must be pixel-aligned to RGB** |
| Dtype | `float32` |
| Units | **meters** (metric) |
| Missing values | `0.0` or `NaN`, both supported; mapped to `valid_mask = 0` |
| Convention | Depth along optical axis (= z in camera frame), not ray length |
| Range | Reasonable indoor range 0.3–10 m; out-of-range values trigger a warning but are kept |
| Letterbox handling | Depth is resized and letterbox-padded with the same transform as RGB; padded regions get `valid_mask = 0` |

**Critical:** if your sensor reports depth in millimeters (most do), divide by 1000 before passing to the model. A pre-flight check: if median depth > 100, the model raises an error.

### 2.3 Camera intrinsics `K` — REQUIRED

| Property | Value |
|---|---|
| Shape | `(3, 3)` numpy array |
| Convention | OpenCV pinhole: `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]` |
| Origin | Top-left of the **original** (pre-resize) image |
| Required? | **Yes — no fallback.** |
| After resize | Preprocessing automatically rescales `K` to match the 518×518 letterboxed image. The user passes the original `K`. |

### 2.4 Output schema

```python
[
    {
        "center_m":   np.ndarray,  # (3,) float32, meters, camera-frame
        "size_m":     np.ndarray,  # (3,) float32, meters; order (w, l, h)
        "R":          np.ndarray,  # (3, 3) float32, SO(3), camera-frame
        "corners_m":  np.ndarray,  # (8, 3) float32; derived for convenience
        "class_id":   int,
        "class_name": str,
        "score":      float,       # in [0, 1]
    },
    ...
]
```

Sorted by `score` descending. Empty if no detection passes `score_threshold`.

---

## 3. System diagram

```
        ┌──────────────────────────────────────────────────────┐
        │  INPUT:                                              │
        │   • 1 RGB image (518×518 after preprocess)           │
        │   • 1 depth map (m, aligned to RGB)                  │
        │   • 1 intrinsics K (3×3) — REQUIRED                  │
        └──────────────────────────────────────────────────────┘
                                 │
            ┌────────────────────┴────────────────────┐
            ▼                                         ▼
 ┌───────────────────────┐               ┌─────────────────────────┐
 │ Depth → Point Cloud   │               │  MapAnything @ N=1      │
 │ unprojection          │               │  (frozen, features-only)│
 │                       │               │                         │
 │  for each pixel (u,v):│               │  Outputs we USE:        │
 │   if depth(u,v) valid:│               │   • enc_feat (2D head)  │
 │     z = depth(u,v)    │               │   • aat_feat[15] (memo) │
 │     x = (u-cx)·z/fx   │               │  Outputs we IGNORE:     │
 │     y = (v-cy)·z/fy   │               │   • pts3d (predicted)   │
 │     pts3d[v,u]=[x,y,z]│               │   • depth_along_ray     │
 │   valid_mask[v,u]=    │               │   • cam_*, metric_scale │
 │     depth_valid       │               │   • aat_feat[7], [11]   │
 │                       │               │                         │
 │  No learnable weights.│               │  (config flag: skip     │
 │  Deterministic.       │               │   geometric heads at    │
 │                       │               │   inference)            │
 │  pts3d ∈ ℝ^(518,518,3)│               │                         │
 │  valid_mask ∈ {0,1}   │               │                         │
 │           ^^^^^^^^^^^ │               │                         │
 │           per-pixel   │               │                         │
 └───────────────────────┘               └─────────────────────────┘
            │                                         │
            └────────────────────┬────────────────────┘
                                 ▼
 ┌────────────────────────────────────────────────────────────────┐
 │  Single-layer memory                                           │
 │   • aat_feat[15] → 1×1 proj 1536 → 512                         │
 │   • 1369 tokens (37 × 37)                                      │
 │   • Per-token PE3D from depth-derived pts3d at token's pixel   │
 │     (use learnable missing_pe when valid_mask=0 at that pixel) │
 │   • NO level-id embedding (single level)                       │
 └────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
 ┌─────────────────┐    ┌────────────────────────────────────────┐
 │ 2D Detection    │───▶│  Query Generator (top-K=256)           │
 │ Head            │    │   For each 2D box (uᵢ,vᵢ,score,class): │
 │ (4-layer DETR,  │    │    1. Snap (uᵢ,vᵢ) to nearest pixel    │
 │  256-dim,       │    │       with valid_mask=1 inside box      │
 │  100 queries,   │    │       (fallback: nearest valid pixel    │
 │  reads enc_feat)│    │       within ±20% box dilation)         │
 └─────────────────┘    │    2. p̂ᵢ = pts3d at snapped pixel       │
                        │    3. fᵢ = RoIAlign(aat_feat[15], box) │
                        │    4. eᵢ = Fourier-PE-3D(p̂ᵢ)            │
                        │    5. qᵢ = MLP([proj(f); e; sc; cls])  │
                        │   If no valid pixel anywhere in box:    │
                        │       drop the query (don't fallback)   │
                        └────────────────────────────────────────┘
                                 │
                                 ▼
 ┌────────────────────────────────────────────────────────────────┐
 │  3D Decoder × 8 layers (DETR, pre-norm, 512-dim, FFN 512)      │
 │   q = SelfAttn(q + PE3D(current center))                       │
 │   q = CrossAttn(q, memory + memory_PE)                         │
 │   q = FFN(q)                                                   │
 │   Deep supervision: aux head at every layer                    │
 │   Cross-attn cost: 256 × 1369 × 512 (~4× cheaper than          │
 │   a multi-scale baseline)                                       │
 └────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
 ┌────────────────────────────────────────────────────────────────┐
 │  3D Heads (per query, per decoder layer)                       │
 │   cls_logits ∈ ℝ³¹   (C + 1 background; C = 30)                │
 │   Δcenter   ∈ ℝ³                                               │
 │   log_size  ∈ ℝ³                                               │
 │   rot6d     ∈ ℝ⁶                                               │
 │   log_σ²    ∈ ℝ⁷   (V3 only)                                   │
 └────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
 ┌────────────────────────────────────────────────────────────────┐
 │  Postprocess: top-K, optional 3D NMS @ 0.25                    │
 │  Output: list of dicts (§2.4)                                  │
 └────────────────────────────────────────────────────────────────┘
```

---

## 4. Module spec

### 4.1 MapAnything backbone (features-only, N=1)

**What it does.** Provides visual features at multiple scales. *Nothing else.*

**Inputs to MapAnything.**
```python
views = [{
    "img":               rgb_tensor,            # (1, 3, 518, 518), float32, DINOv2-normalized
    "data_norm_type":    "dinov2",
    "depth_along_ray":   depth_tensor,          # (1, 518, 518, 1), meters — PRIOR ONLY
    "is_metric_scale":   torch.tensor([True]),
    "ray_directions_cam": ray_dirs,             # (1, 518, 518, 3), computed from K
}]
```

> **Why we still feed depth as a prior even though we don't use MapAnything's `pts3d` output.** Feeding depth conditions the AAT layers to produce more geometrically consistent *features*. The MapAnything paper (Table 4 monocular depth completion ablation) reports that features improve when depth is provided as a prior — even when only depth output is evaluated. We piggyback on this for free.

**Outputs we use (and discard).**

| Key | Shape | Status |
|---|---|---|
| `enc_feat` | (37, 37, 1536) | ✓ Used by 2D head (§4.4) |
| `aat_feat[15]` | (37, 37, 1536) | ✓ Used by 3D head's single-layer memory (§4.3) |
| `aat_feat[7]`, `aat_feat[11]` | (37, 37, 1536) each | ✗ Computed but discarded |
| `pts3d` (predicted) | (518, 518, 3) | ✗ Ignored — we use depth-unprojected pts3d instead |
| `depth_along_ray` | (518, 518, 1) | ✗ Ignored |
| `cam_quats`, `cam_trans` | (4,), (3,) | ✗ Identity at N=1 |
| `metric_scaling_factor` | scalar | ✗ Ignored |

**Frozen.** Yes. No backbone fine-tuning in this variant.

**Inference compute saving.** Because we ignore most outputs, the prediction heads for `pts3d` / depth / cam can be skipped at inference via config flag `backbone_geometry_heads=False`, saving ~5–10% backbone forward time. At training time we keep them enabled because gradients through them stabilize the AAT (they were jointly trained).

### 4.2 Depth-to-point-cloud unprojection

**What it does.** Deterministic geometric step that turns the depth map + `K` into a per-pixel 3D point cloud in the camera frame.

**Operation.**
```python
def unproject_depth(depth, K, valid_mask=None):
    """
    depth : (H, W) tensor, meters
    K     : (3, 3) tensor, OpenCV intrinsics for this image
    Returns:
        pts3d      : (H, W, 3) tensor, camera frame
        valid_mask : (H, W) bool tensor
    """
    H, W = depth.shape
    if valid_mask is None:
        valid_mask = (depth > 0) & torch.isfinite(depth)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    v, u = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    u, v = u.float(), v.float()
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts3d = torch.stack([x, y, z], dim=-1)
    pts3d = torch.where(valid_mask[..., None], pts3d, torch.zeros_like(pts3d))
    return pts3d, valid_mask
```

**Cost.** Trivially cheap (linear in pixel count). Fully differentiable but we don't backprop into depth — it's an input.

**`K` rescaling for letterbox resize.** The user passes `K` for the original image. Preprocessing resizes to 518×518 with letterbox; the same affine rescale is applied to `K` so `unproject_depth` uses K matching the resized depth map. The utility `rescale_K_for_letterbox` (in `mapdet3d_mono_depthpc_last.preprocess`) handles this.

**Per-pixel granularity.** Output `pts3d` is at full 518×518 resolution. The token-grid PE (§4.3) samples this at the token's pixel center via bilinear interpolation; the query generator (§4.5) samples at the chosen anchor pixel.

**Valid mask handling.** Propagated through the rest of the pipeline:
- §4.3 (memory tokens): tokens with `valid_mask = 0` at the token's pixel get a learnable "no-3D" PE instead of `PE3D([0,0,0])` (which would be a confidently-wrong "you're at the camera origin" signal).
- §4.5 (query generator): 3-tier snap-to-valid fallback.

### 4.3 Single-layer feature memory

The 3D decoder's cross-attention memory is built from one feature map: MapAnything's final AAT layer.

| Step | Detail |
|---|---|
| Input feature map | `aat_feat[15]` of shape `(1, 37, 37, 1536)`, bfloat16 |
| Per-feature projection | One 1×1 conv: 1536 → 512 |
| Flatten to tokens | 37 × 37 × 1 = **1369 tokens** |
| Per-token feature | `f_t ∈ ℝ^512` |
| Per-token 3D-PE | `PE3D(pts3d_at_token_pixel)` if `valid_mask_at_token_pixel = 1`, else **`missing_pe`** (learnable 512-dim vector) |
| Level-id embedding | **None** (single level — there's nothing to distinguish) |
| Final memory | `(1, 1369, 512)` |
| Final memory PE | `(1, 1369, 512)` |

**Pixel-to-token correspondence.** Each token corresponds to a 14×14 image patch. Sample `pts3d` at the *center pixel* via bilinear interpolation, `valid_mask` via nearest-neighbor. A token is considered "valid" if its center pixel has `valid_mask = 1`.

**Why a learnable `missing_pe` instead of dropping invalid tokens.** Dropping changes the token count per-batch (variable-length), which is annoying for batched attention. Keeping all 1369 tokens with a special "missing" PE lets the decoder learn to ignore them via attention — at the cost of slightly wasted compute on dead tokens. Indoor scenes typically have ≤5% missing pixels from modern depth sensors, so the waste is small.

**Why single-layer memory (the variant's defining design).** MapAnything's 16 AAT layers operate at a constant 37×37 spatial grid — no downsampling. The standard argument for multi-scale fusion (FPN-style C2/C3/C4/C5 with different spatial resolutions providing complementary information) doesn't directly apply when spatial resolution doesn't change across taps. The hypothesis is that 16 layers of joint geometric refinement (AAT[15]) already integrate everything useful from earlier taps. See §6.5 for the full reasoning.

**Compute saving.** With 256 queries × 1369 keys, the 3D decoder's cross-attention is ~25% of what it would be against a multi-scale memory (5476 keys). End-to-end inference: ~130 ms vs. ~150 ms on H100. Peak training memory: ~38 GB vs. ~55 GB per GPU at batch 2 × 16.

### 4.4 2D detection head

**Why we have a 2D head.** WildDet3D's published ablation shows that removing a jointly-trained 2D head and predicting 3D directly drops AP from 30.2 → 11.1, a **−19.1 AP** loss. The 2D loss provides dense, low-noise supervision that anchors the 3D head's center/size regression in image space, and the 2D head's predictions become the 3D head's query anchors (§4.5).

**Why a small head on shared DINOv2 features (not OWLv2/SAM3).** Our target is closed-vocab. OWLv2/SAM3 are open-vocab, carrying a text encoder we wouldn't use and a separate backbone that doubles the FLOPs. A small head on top of MapAnything's already-running DINOv2 encoder is strictly better.

**Module shape.**
- Input: `enc_feat` (37, 37, 1536). The 2D task doesn't need geometric refinement, so we don't use the AAT taps for the 2D head.
- Decoder: 4 layers, 256-dim, 4 heads, pre-norm DETR-style.
- 100 learnable queries per image. Indoor images typically contain 5–15 objects, so 100 is over-provisioned.
- Heads: classifier (31 logits = 30 classes + background, focal loss) + 2D-box regressor (4 outputs: `cx, cy, w, h` in normalized image coords, L1 + GIoU2D).
- Parameter count: ~10M.

**Hungarian matching.** Standard DETR cost: `λ_cls · focal(cls) + λ_l1 · L1(box) + λ_giou · GIoU2D(box)`.

### 4.5 Query generator (depth-PC anchored, snap-to-valid)

For each of the top-K (K = 256) 2D boxes by classification score, produce a 3D query, falling back gracefully on depth holes:

```python
def make_query(box_2d, score, pred_cls, pts3d, valid_mask, feat_map_aat15):
    """
    Returns (query, query_pe, anchor_p) or None if no valid pixel anywhere in box.
    """
    # Step 1: snap to a valid pixel
    anchor_uv = snap_to_valid_pixel(box_2d, valid_mask)
    if anchor_uv is None:
        return None  # depth missing everywhere in the box; drop this query

    # Steps 2-5: build the query
    p3d       = bilinear_sample(pts3d, *anchor_uv)
    roi_feat  = roi_align(feat_map_aat15, box_2d)
    pos_enc   = fourier_pe_3d(p3d, dim=512)
    score_emb = nn.Linear(1, 32)(score)
    cls_emb   = nn.Embedding(num_classes, 64)(pred_cls)
    query     = MLP([proj_512(roi_feat); pos_enc; score_emb; cls_emb])
    return query, pos_enc, p3d
```

**`snap_to_valid_pixel` algorithm — 3-tier fallback:**
```python
def snap_to_valid_pixel(box_2d, valid_mask):
    cx, cy = box_center(box_2d)
    # Tier 1: the geometric center
    if valid_mask[cy, cx]:
        return (cx, cy)
    # Tier 2: nearest valid pixel inside the 2D box
    box_region = valid_mask[ymin:ymax, xmin:xmax]
    if box_region.any():
        return nearest_valid_pixel_to_center(box_region, (cx, cy))
    # Tier 3: nearest valid pixel within a 20%-dilated box
    dilated = dilate_box(box_2d, factor=1.2)
    box_region = valid_mask[dilated_slice]
    if box_region.any():
        return nearest_valid_pixel_to_center(box_region, (cx, cy))
    return None  # give up
```

**Three-tier fallback rationale.**
- **Tier 1** (geometric center): handles the easy case — depth is dense, center pixel is valid.
- **Tier 2** (any valid in box): handles small holes (e.g., a glossy patch on a desk surface). We snap to whatever depth we can get.
- **Tier 3** (20% dilation): handles boxes that fall entirely on a depth-hole (e.g., a glass door whose interior is all NaN). We snap to a valid pixel just outside, accepting that the 3D anchor is approximate.
- **Drop** the query if even tier-3 fails. Better to miss a detection than anchor a query at `[0,0,0]` (which the decoder would interpret as "object at the camera origin" — confusing).

**RoIAlign feature source.** `aat_feat[15]` directly — no blend with `enc_feat`. The 3D-head's memory comes from `aat_feat[15]` only, so the query's RoI features should come from the same source to avoid a distribution mismatch the decoder would otherwise have to absorb.

**Fallback FPS.** If total valid queries < K = 256, use attention-guided FPS over the valid-only point cloud (weighted by the 2D classifier's score map) to fill remaining slots.

### 4.6 3D decoder

**Architecturally unchanged across variants:** 8 layers, 512-dim, 4 heads, FFN dim 512, dropout 0.1, pre-norm. Self-attention with iteratively-refined anchor PE; cross-attention to the 1369-token memory; deep supervision at every layer (aux head).

What changes from a multi-scale baseline is *what the decoder sees*: 1369 keys instead of 5476, so cross-attention cost is ~25% of the baseline. Decoder weights have the same shape but won't transfer cleanly across memory configurations (they were trained against different memory distributions).

### 4.7 Box parameterization and prediction heads

Per query, per decoder layer, the heads output **12 + (C + 1) = 43 dims** (with `C = 30`):

| Output | Dim | Meaning |
|---|---|---|
| `cls_logits` | 31 (= C+1) | 30 named classes + 1 background |
| `Δcenter` | 3 | offset from anchor `p̂ᵢ` to predicted center, meters |
| `log_size` | 3 | log-scale (w, l, h), meters |
| `rot6d` | 6 | first two rows of 3×3 rotation matrix; R recovered by Gram-Schmidt |
| `log_σ²` (V3 only) | 7 | aleatoric log-variance for the 7 box dimensions |

**Box decoding (eval-time):**
```python
center  = anchor_p + Δcenter        # meters
size    = exp(log_size)             # meters
R       = gram_schmidt(rot6d)       # SO(3)
corners = center + R @ diag(size) @ unit_cube_corners   # (8, 3)
```

**Canonical normalization at GT-loading time.** Enforce `w ≤ l` and yaw ∈ `[0, π)` on GT to eliminate a 4-fold yaw symmetry that would otherwise make the rotation loss bimodal.

### 4.8 Losses and matching

```
L_total = L_2D + L_3D

L_2D  = λ_2D_cls·FocalLoss(cls2d, y_cls)
      + λ_2D_l1 ·L1(pred_2dbox, gt_2dbox)
      + λ_2D_giou·(1 − GIoU2D(pred_2dbox, gt_2dbox))

L_3D  = Σ_{layer L ∈ 1..8} (
          λ_cls·FocalLoss(cls_logits, y_cls)
        + λ_ctr·L1(pred_center, gt_center)              # meters
        + λ_sz ·L1(log_size, log(gt_size))
        + λ_rot·‖R_pred − R_gt‖_F                       # chord distance
        + λ_box·MGIoU3D(pred_corners, gt_corners)
        + λ_unc·NLL(pred, gt, σ²)                       # V3 only
        + λ_dn ·DenoisingLoss(...)                      # V3 only
      )
```

**Default weights.**
- 2D head: `λ_2D_cls = 2, λ_2D_l1 = 5, λ_2D_giou = 2`.
- 3D head: `λ_cls = 2, λ_ctr = 5, λ_sz = 1, λ_rot = 2, λ_box = 2, λ_unc = 0.1, λ_dn = 1`.

**Hungarian matching cost (assignment, not backprop):** `2·focal + 5·L1(center) + 2·MGIoU3D + 0.5·L1(log_size)`. Optionally one-to-many above MGIoU threshold 0.1 (default: enabled).

**MGIoU 3D (brief).** Project both predicted and GT 8-corner boxes onto 6 candidate face-normal axes, compute 1D GIoU per axis, average. Smooth everywhere, rotation-invariant, pure PyTorch (10–40× faster than exact rotated 3D IoU per the paper).

**MGIoU warmup.** For the first 2k optimizer steps, set `λ_box = 0` and use L1-on-corners. After 2k steps, linearly ramp `λ_box: 0 → 2` over 5k more steps.

**`missing_pe` L2 regularization.** The learnable `missing_pe` vector (§4.3) gets ~4× less gradient signal in this variant than in a multi-scale baseline (fewer total tokens, only a fraction invalid). To prevent the vector from drifting under the small gradient signal, we apply an L2 prior with weight `λ_missing_pe_l2 = 0.05` (5× tighter than would be reasonable for a multi-scale baseline). If you observe the vector under-trains, walk it back to 0.01.

**Queries dropped by the snap-to-valid fallback are not fed to the decoder.** The Hungarian matching for that scene is over `K_eff ≤ K = 256` queries. Track per-batch in the loss reducer.

**Gradient balancing.** Monitor `||∇_2D|| / ||∇_3D||` at the shared encoder; if it leaves `[0.5, 2.0]`, GradNorm (Chen et al. 2018) rescales `λ_2D_*` automatically.

---

## 5. Coordinate conventions and unit handling

| Concept | Convention |
|---|---|
| Image coords | (u, v), top-left origin, u right, v down |
| Camera frame | x right, y down, z forward (OpenCV) |
| Box center | Camera frame, meters |
| Box size | `(w, l, h)` after R applied |
| Rotation R | R ∈ SO(3); `corner = center + R @ diag(size) @ unit_corner` |
| Depth | Meters along optical axis |
| K | OpenCV pinhole, top-left origin |
| Output frame | Camera frame |

**Unit pitfalls.**
- Depth in mm → divide by 1000.
- `K` for the original image, *not* the resized one (preprocessing rescales).
- Y axis points DOWN in OpenCV (gravity is `+y`); a chair on the floor in front of the camera has `center.y > 0` and `center.z > 0`.

---

## 6. Why these choices (intuition + evidence)

### 6.1 Why unproject depth instead of using a learned pts3d

- **Intuition.** When you have a calibrated metric depth sensor, the sensor's 3D points are usually more accurate than a network's learned predictions — by construction. Unprojection is a one-line operation with no learned parameters and no failure modes beyond bad depth itself.
- **Evidence.** Sensor depth from modern ToF / structured-light / iPad LiDAR is typically accurate to <2 cm at indoor distances. Learned pts3d, while strong, is still a regression target with non-zero error and occasional structured failures (mirrors, transparent surfaces, very dark materials). Trusting the sensor is the conservative, deployment-safe choice.
- **Risk if wrong.** Bad/sparse depth makes unprojected pts3d worse than a learned alternative. This variant is *for* the case where depth is trusted; deployers with poor depth should pick a sibling variant.

### 6.2 Why still use MapAnything for features (not pure DINOv2)

- **Intuition.** MapAnything's encoder *is* DINOv2 ViT-G, so we're not adding a separate model — the encoder runs anyway. The AAT layers on top are pretrained on the geometric task and produce features that are more 3D-consistent than vanilla DINOv2 (adjacent pixels with similar 3D positions get similar features). This is a free bonus for the 3D head's cross-attention.
- **Evidence.** Empirically the AAT adds value to MapAnything's downstream tasks (paper Table 4 ablation of "no info-sharing").
- **Risk if wrong.** AAT features are no better than encoder-only features. Mitigation: a future variant with `aat_feat[7,11]` dropped and only encoder + aat_final fused (a 2-level fallback). If that performs as well, ship a lighter "dinov2-only" variant.

### 6.3 Why feed depth as a prior to MapAnything even though we ignore the geometric output

- **Intuition.** MapAnything's AAT was trained with depth as an optional input — when depth is provided, the AAT internally generates better-conditioned *features* (regardless of whether we read the geometric output). Free improvement.
- **Evidence.** MapAnything paper §3 describes the per-view input augmentation scheme where depth is fed with probability 0.95 during training.
- **Risk if wrong.** Negligible. Cost is zero. If it hurts somehow, flip it off via config flag.

### 6.4 Why "snap to valid pixel" with a 3-tier fallback

- **Intuition.** Real depth sensors have holes — small ones on glossy surfaces, larger ones on glass/mirrors. A naïve `pts3d_at(box_center)` would silently anchor queries at `(0, 0, 0)` for those pixels — confusing the decoder. Snapping to the nearest valid pixel keeps the anchor approximately right; dropping the query as a last resort beats anchoring at the origin.
- **Evidence.** Standard practice in RGB-D processing (Open3D, MMDetection3D's depth-image input pipelines).
- **Risk if wrong.** We drop too many queries when depth is sparse. Mitigation: monitor `K_eff / K` during training — if < 0.7 on average, the dataset has too much missing depth and the variant isn't appropriate.

### 6.5 Why drop the multi-scale fusion (the variant's other defining choice)

**Hypothesis.** AAT[15] is the result of 16 transformer layers operating jointly on geometric features. By design, the final layer integrates information from all earlier layers via residual connections. For a downstream task (3D box prediction) that needs geometry-aware semantic features, AAT[15] should already contain everything useful from AAT[7] and AAT[11], plus more.

**Intuition.** In standard object detection backbones with FPN, multi-scale fusion is essential because the backbone downsamples spatially as it goes deep — C2/C3/C4/C5 have different spatial resolutions and complementary information. MapAnything's AAT does **not** downsample: all 16 AAT layers operate at the same 37×37 token grid. The case for multi-scale fusion is weaker when spatial resolution doesn't change across taps.

**Evidence (mixed).**
- **Against the variant:** VGGT-Det (CVPR 2026) found multi-layer > last-only on VGGT's similar architecture (their §3.1 ablation). Most relevant prior.
- **For the variant:** Many transformer detection heads (DETR with single-scale ViT memory, DAB-DETR with single-scale Swin) work well without multi-scale fusion. The single-scale story is well-precedented when the backbone is strong.
- **Honest position.** We don't know which one wins on our specific architecture. The variant exists to answer this question. Expected small AP gap (1–3 mAP@0.25) in favor of multi-scale, traded for ~15% inference time + ~30% peak training memory savings. If the gap is < 2 mAP, this variant is the better deployment choice.

**Risk if wrong.** If this variant loses > 3 mAP@0.25 vs. a multi-scale baseline, the multi-scale design is providing real value. Mitigation: fall back to 2-level fusion (`enc_feat` + `aat_feat[15]`), config-only change.

### 6.6 Why a learnable "missing-PE" for memory tokens with missing depth

- **Intuition.** Memory tokens that correspond to missing-depth pixels would otherwise get `PE3D([0,0,0])` — a confidently-wrong "you're at the camera origin" signal. A dedicated learnable vector lets the decoder explicitly mask attention to those tokens.
- **Evidence.** Standard trick (BEVFormer, PETR) for "off-grid" tokens.
- **Risk if wrong.** The learnable vector trains to a non-trivial value that interferes with attention, especially given the smaller gradient signal in this variant (fewer total tokens). Mitigation: initialize to a near-zero vector and apply an L2 prior with weight 0.05 (5× the obvious default) to keep it small. Walk back to 0.01 if under-training is observed.

### 6.7 Why 6D rotation instead of quaternions

- **Intuition.** Quaternions have an antipodal sign ambiguity (q ≡ -q) that creates a discontinuity in the loss landscape. The 6D representation (Zhou et al. 2019) is the smallest representation guaranteed continuous over all of SO(3).
- **Evidence.** Zhou et al. CVPR 2019. Standard in modern 3D detection.

---

## 7. Tensor shape reference

For one inference call (B = 1, N = 1):

| Stage | Tensor | Shape | Dtype |
|---|---|---|---|
| Input | `rgb` | (1, 3, 518, 518) | float32 |
| Input | `depth` | (1, 518, 518) | float32 |
| Input | `K` | (1, 3, 3) | float32 |
| Unprojection | `pts3d` | (1, 518, 518, 3) | float32 |
| Unprojection | `valid_mask` | (1, 518, 518) | bool |
| MapAnything | `enc_feat` | (1, 37, 37, 1536) | bfloat16 |
| MapAnything | `aat_feat[15]` | (1, 37, 37, 1536) | bfloat16 |
| MapAnything | `aat_feat[7]`, `aat_feat[11]` (computed, discarded) | (1, 37, 37, 1536) each | bfloat16 |
| MapAnything | geometric heads (not computed if config flag set) | — | — |
| Memory | projected + flattened | (1, 1369, 512) | float32 |
| Memory PE | with `missing_pe` for invalid tokens | (1, 1369, 512) | float32 |
| 2D head | `det2d_outputs` | (1, 100, 35) — 31 cls + 4 box | float32 |
| Query gen | `q`, `q_pe`, `anchor_p` | (1, K_eff, 512), (1, K_eff, 512), (1, K_eff, 3) | float32 |
| Query gen | `K_eff` | scalar ≤ 256 | int |
| 3D decoder | `q_out` per layer | (1, K_eff, 512) | float32 |
| 3D heads | `cls_logits` | (1, 8, K_eff, 31) | float32 |
| 3D heads | `pred_center` | (1, 8, K_eff, 3) | float32 |
| 3D heads | `pred_corners` | (1, 8, K_eff, 8, 3) | float32 |

---

## 8. Risks, failure modes, and mitigations

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| User passes depth in mm not m | H | H | Pre-flight check (median depth > 100 → error). |
| User passes K for resized image, not original | M | H | Preprocessing rescales for them; warn if K[0,0] < 100 ("Does your K look right?"). |
| Depth and RGB are not pixel-aligned | M | H | Document the alignment requirement; provide a sample warp utility for common sensors. |
| Depth has > 30% missing pixels | M | M | `K_eff / K` ratio logged at inference; warning if < 0.7. |
| Depth is correct but `K` is wrong by ~10% | M | H | Unprojected pts3d wrong by the same factor — silent failure. At inference, if MapAnything's predicted K differs from user's K by > 10%, log a warning. |
| Snap-fallback's 3-tier rule fails on entire-box-on-glass cases | L | M | Query is dropped (intended). Downstream code handles `K_eff < K`. |
| `missing_pe` vector trains too slowly because of fewer total tokens AND only a fraction invalid | M | M | Tighter L2 prior (`λ = 0.05`). Monitor `||missing_pe||` — if it stays much smaller than other memory PE vectors' norms by mid-S2, walk the L2 back to 0.01. |
| Combined "single-layer + depth-unprojection" interacts worse than either change alone | M | H | Ablate against single-changes individually. If combined < min(single-change variants) by > 1 mAP, ship a single-change variant instead. |
| Multi-scale fusion was providing real signal — small-object AP regresses | M | M | Per-class AP for `{lamp, monitor, pillow, tv}`. If they drop > 5 mAP relative to a multi-scale baseline, fall back to 2-level fusion. |
| MGIoU flat on degenerate predictions early in training | M | M | L1 warmup + loss-weight ramp. |
| 2D / 3D gradient imbalance | M | M | GradNorm auto-rebalancing. |
| Canonical rotation normalization mis-handled at eval | M | H | Unit-test on 100 synthetic boxes before training; visualize first 100 GT boxes per dataset. |

---

## 9. Pseudocode (single inference step)

```python
import torch
from mapdet3d_mono_depthpc_last import MapDet3DMonoDepthPCLast, preprocess

model = MapDet3DMonoDepthPCLast.from_pretrained(
    "mapdet3d-mono-depthpc-last-v1"
).cuda().eval()

# 1. Preprocess: letterbox resize, normalize RGB, rescale K, prepare ray_directions
batch = preprocess(rgb_np, depth_np, K_np)

with torch.no_grad(), torch.autocast(dtype=torch.bfloat16):
    # 2. Unproject depth — deterministic, no learnable weights
    pts3d, valid_mask = model.unproject_depth(batch["depth_raw"], batch["K"])

    # 3. Backbone — features only; geometric outputs discarded
    enc_feat, aat_feats = model.backbone.features_only(batch)
    aat15 = aat_feats[-1]  # only the final layer is used downstream

    # 4. 2D head
    det2d = model.head_2d(enc_feat)

    # 5. Build queries (some may be dropped due to depth holes)
    q, q_pe, anchor_p, k_eff = model.query_generator(
        pts3d=pts3d, valid_mask=valid_mask,
        aat_final=aat15,
        det2d=det2d, K=256,
    )

    # 6. Build single-layer memory with valid-mask-aware PE
    mem, mem_pe = model.build_memory(aat15, pts3d, valid_mask)

    # 7. 3D decoder + heads (use layer-8 output for prediction)
    q_out = model.decoder(q, mem, q_pe, mem_pe)[-1]
    cls_logits, center_pred, log_size, rot6d = model.head_3d(q_out, anchor_p)

    # 8. Decode boxes
    boxes = model.decode_boxes(
        cls_logits, center_pred, log_size, rot6d, anchor_p,
        score_threshold=0.3, nms_iou=0.25,
    )
```

---

## 10. References

| # | Paper / Repo | What we use it for |
|---|---|---|
| 1 | Keetha et al., *MapAnything*, 3DV 2026. [arXiv:2509.13414](https://arxiv.org/abs/2509.13414) | Feature backbone (encoder + AAT, geometric outputs unused) |
| 2 | Xu et al., *VGGT-Det*, CVPR 2026. [arXiv:2603.00912](https://arxiv.org/abs/2603.00912) | Decoder template; multi-layer-vs-last ablation we explicitly test the opposite of |
| 3 | Le et al., *MGIoU*, AAAI 2026 Oral. [arXiv:2504.16443](https://arxiv.org/abs/2504.16443) | 3D box loss |
| 4 | Huang et al., *WildDet3D*, 2026. [arXiv:2604.08626](https://arxiv.org/abs/2604.08626) | Training data, 12-dim encoding, joint 2D-head evidence |
| 5 | DeTone et al., *Boxer*, Meta Reality Labs, 2026. [arXiv:2604.05212](https://arxiv.org/abs/2604.05212) | 2D-to-3D query lifting recipe |
| 6 | Yang et al., *3D-MOOD*, ICCV 2025. [arXiv:2507.23567](https://arxiv.org/abs/2507.23567) | End-to-end 2D + 3D joint training precedent |
| 7 | Misra et al., *3DETR*, ICCV 2021 | Decoder, query PE (Fourier 3D) |
| 8 | Zhou et al., *On the Continuity of Rotation Representations*, CVPR 2019 | 6D rotation representation |
| 9 | Liu et al., *PETR*, ECCV 2022 | 3D-position-on-image-tokens; missing-PE trick |
| 10 | Li et al., *BEVFormer*, ECCV 2022 | Learnable "off-grid" token PE precedent |
| 11 | Chen et al., *GradNorm*, ICML 2018 | 2D/3D loss auto-balancing |
| 12 | Brazil et al., *Omni3D / Cube R-CNN*, CVPR 2023 | Closed-vocab indoor 3D detector precedent |
| 13 | Roberts et al., *Hypersim*, ICCV 2021 | Training dataset |
| 14 | Baruch et al., *ARKitScenes*, NeurIPS 2021 | Zero-shot eval target |

---

## 11. Glossary

- **AAT** — Alternating-Attention Transformer. MapAnything's cross-view fusion module. At N=1 it reduces to self-attention on a single view but still refines features through 16 transformer layers at a constant 37×37 grid.
- **C** — Number of named foreground classes. Fixed at 30.
- **DINOv2** — Self-supervised pre-trained ViT. MapAnything's encoder is DINOv2 ViT-G/14.
- **DETR / DETR-style** — End-to-end transformer detector with learnable queries + Hungarian matching.
- **Depth-PC** — Point cloud derived by unprojecting a depth map with `K`. Deterministic; no learnable parameters.
- **GIoU2D** — Generalized IoU for 2D boxes.
- **GradNorm** — Multi-task loss-weight auto-balancing algorithm.
- **K** — 3×3 camera intrinsic matrix (pinhole).
- **K_eff** — Effective number of queries after some are dropped due to depth holes (§4.5).
- **Letterbox** — Image-resize technique that preserves aspect ratio by padding.
- **MGIoU** — Marginalized Generalized IoU; smooth, rotation-invariant 3D box loss.
- **missing_pe** — Learnable 512-dim vector used as the 3D positional encoding for memory tokens whose pixel has no valid depth (§4.3).
- **NYU40** — A 40-class indoor semantic taxonomy; our 30-class set is a curated subset.
- **OBB** — Oriented Bounding Box.
- **OpenCV camera convention** — (x right, y down, z forward); top-left image origin.
- **PE3D** — 3D positional encoding (Fourier sin/cos at multiple frequencies).
- **PETR** — Position-Embedded Transformer (CVPR 2022) — coined the per-token 3D-PE trick.
- **pts3d** — Per-pixel 3D point cloud. In this variant, computed by depth-unprojection.
- **Snap-to-valid** — The 3-tier query-anchor pixel selection rule for handling depth holes (§4.5).
- **Unprojection** — The geometric operation `[x, y, z] = ((u - cx)/fx, (v - cy)/fy, 1) · z` (per-pixel; uses `K` and `depth`).
- **valid_mask** — Boolean per-pixel tensor indicating whether the depth is valid (non-zero, finite, in-bounds after letterbox).
- **ViT-G** — Vision Transformer at "Giant" scale.

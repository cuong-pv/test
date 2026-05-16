# MapDet3D — Architecture & Research Design (v3)

**A MapAnything-Based 3D Object Detection Head — jointly trained on Hypersim + WildDet3D, zero-shot on ARKitScenes**

| | |
|---|---|
| **Author** | Cuong Pham |
| **Date** | 2026-05-16 |
| **Status** | Design proposal (pre-implementation), v3 — scope: Hypersim + WildDet3D primary, ARKit zero-shot |
| **Working name** | MapDet3D (placeholder) |
| **Reference repos** | `VGGT-Det-CVPR2026`, `map-anything`, `MGIoU`, `WildDet3D`, `Boxer`, `3detr` (all local in `/3DDET/`) |

**Convention used throughout this doc.** Every concrete architectural decision is presented as:
> **Choice.** What we will do. **Intuition.** Why we expect it to work. **Evidence.** A specific paper/section or `file:line` citation. **Risk if wrong.** What we'd see in training metrics if the assumption fails, and the fallback.

This makes the doc auditable: any reviewer (including future-you) can disprove a claim by checking the evidence pointer.

---

## 0a. Changelog (v2 → v3)

| Change | Reason | Sections affected |
|---|---|---|
| **Primary targets changed** to Hypersim + WildDet3D (jointly trained); ARKitScenes demoted to zero-shot eval | User decision | §1, §2, §5, §6, §7, §10 |
| **Dropped OWLv2 / SAM3** (open-vocab) from the 2D detector slot — inconsistent with closed-vocab target. Replaced with a **jointly-trained closed-vocab 2D head** sharing MapAnything's DINOv2 encoder (3D-MOOD / RF-DETR-style). | User feedback: closed-vocab means we shouldn't pay for open-vocab machinery | §4.2.1, §4 diagram, §8 risk row, §9 timeline, §10 Q4 |
| Added **unified closed-vocab taxonomy** (§6.0) — Hypersim's NYU40-style classes ⊔ WildDet3D's 13.5k flattened to a ~60-class indoor set | Required to train one model on two label spaces | §6.0 (new) |
| Added **WildDet3D-grounded training recipe** (stage curriculum, dataset mixing, freezing) | Their published recipe is the strongest available prior on multi-dataset 3D detection training | §7 |
| Added **ARKit zero-shot eval protocol** with class-mapping table | New evaluation requirement | §6.3, §10 |
| Removed Boxer dataset-overlap warning (no longer relevant — ARKit isn't a training target) | Scope change | §3.4 |
| Updated success criteria for new scope | Hypersim, WildDet3D-Bench, ARKit zero-shot mAP | §2 |

## 0b. Verification log (changes from v1, kept for reference)

The following claims were checked against the actual code and PDFs and corrected:

| Item | v1 (wrong/imprecise) | v2 (verified) | Source |
|---|---|---|---|
| VGGT-Det FFN dim | Suggested 2048 ablation as if it were higher | **512** — `dec_ffn_dim=_token_dim_=512` | `VGGT-Det-CVPR2026/projects/VGGTDet/config/vggtdet_scannet.py:15,53` |
| MapAnything output keys | `camera_pose_quats`, `camera_pose_trans` | **`cam_quats`, `cam_trans`** (the `camera_pose_*` form is the *input* convention) | `map-anything/mapanything/models/mapanything/model.py:1780-1793` |
| ARKitScenes class count | "furniture-focused" only | **17 room-defining furniture categories** | ARKitScenes paper §3, Table 1 |
| Hypersim box format | "per-instance 3D OBBs (7-DoF)" | **9-DoF** (3 position + 3×3 rotation + 3 scale) | `apple/ml-hypersim` README — "tight 9-DOF bounding boxes" |
| Boxer backbone | DINOv2 | **DINOv3** | Boxer paper §3.1; arXiv 2604.05212v1 |
| Boxer training data | "Aria, CA-1M, SUN-RGBD, ScanNet" | Same set **but explicitly excludes ARKitScenes** (overlap with CA-1M) | Boxer paper §4.1 |
| VGGT-Det deep supervision | "assumed" | **Confirmed**: `return_intermediate=True` + `n_levels=8` | `vggtdet.py:820,824`; `config/vggtdet_scannet.py:61` |
| WildDet3D 12-dim encoding | Listed but unsupported | **Verified verbatim** in paper (Δcx/Δcy/10, log d × 2, log w,h,l × 2, r1..r6) | WildDet3D paper §3.2; arXiv 2604.08626v2 |
| WildDet3D 2D-head ablation | "−19.1 AP" | **Confirmed** (30.2 → 11.1) | WildDet3D paper Table 5 |
| MGIoU 10–40× speedup | Cited from web search | **Confirmed in paper PDF** | `papers/2504.16443v2.pdf` Abstract |
| Train views in VGGT-Det | 40 | **42** (the "n_images=42" config) | `config/vggtdet_scannet.py:162` |
| WildDet3D annotation count | "3.9M" | **3.7M verified** (3.9M was an earlier-version count) | WildDet3D paper §3.1 |

---

## 1. Motivation and Problem Statement

VGGT-Det (Xu et al., CVPR 2026; `papers/2603.00912v1.pdf`) showed that a *frozen* multi-view 3D foundation model — VGGT — provides priors strong enough to drive a competitive *sensor-geometry-free* (SG-Free) indoor 3D detector: no calibrated poses, no depth sensor, just RGB. It extracts (a) multi-layer image tokens, (b) predicted depth maps and (c) inferred camera matrices from VGGT and feeds them into a 3DETR-style query decoder with attention-guided FPS query generation. Reported gain over the previous SG-Free SoTA: **+4.4 mAP@0.25 on ScanNet** and **+8.6 mAP@0.25 on ARKitScenes** (paper Abstract).

**Scope (v3).** Unlike VGGT-Det, we target *generalization*: a single model trained jointly on two complementary indoor datasets — Hypersim (synthetic, dense, photorealistic) and WildDet3D (real, in-the-wild, sparser but more diverse) — and evaluated zero-shot on ARKitScenes. The two training datasets are deliberately picked to be opposite ends of the indoor data spectrum: Hypersim is small (46.6k images) but dense (~10 boxes/image, clean GT, controlled lighting); WildDet3D is large (>100k human-annotated indoor images out of its 1M total, ~52% indoor per paper Fig. 6) but noisy (web/in-the-wild, occlusion, variable annotations). The joint regime tests whether MapAnything's metric geometry is strong enough to bridge the synthetic-to-real gap, and whether MGIoU's rotation-invariance is the right loss to absorb both kinds of supervision.

MapAnything (Keetha et al., 3DV 2026; `papers/2509.13414v3.pdf`) is the most recent broad-coverage multi-view foundation model from Meta Reality Labs + CMU. From N input views it directly regresses a **factored metric 3D geometry**: per-pixel depth-along-ray, per-pixel ray maps, camera quaternion + translation, and a global metric scale factor. It uses a DINOv2 ViT-G/14 encoder (first 24 layers) followed by a 16-layer Alternating-Attention Transformer (AAT) for cross-view fusion, giving 40 layers total.

**Why swap the backbone? Intuition.** Three reasons, in decreasing order of expected impact:

1. **Metric output removes one head's worth of work.** VGGT-Det operates in "normalized scene coordinates" — it has to learn the scene scale implicitly from depth, then de-normalize at evaluation time. MapAnything's `metric_scaling_factor` output is, by construction, in meters (verified at `model.py:1690,1783`). The detection head can predict `log_size` and `Δcenter` in physical units from step 0, removing a scale-recovery degree of freedom. This is the dominant expected source of mAP@0.5 gain (tight metric IoU is where VGGT-Det is weakest).
2. **Cross-view fusion is geometry-aware, not just feature-aware.** The AAT alternates frame-wise self-attention with global-across-views attention, with intermediate taps at layers `[7, 11]` plus final (`configs/model/info_sharing/aat_ifr_16_layers_dinov2_vitg_init.yaml:12`). VGGT's aggregation is shallower. We expect more consistent per-view 3D positions at the cross-view fusion stage → better-localized cross-attention in the decoder.
3. **Optional priors can be passed in.** MapAnything accepts `ray_directions_cam`, `depth_along_ray`, `cam_quats`, `cam_trans`, and `is_metric_scale` as *optional* per-view inputs (`model.py:1523-1548`). For indoor RGB-D datasets (ARKitScenes), this lets us optionally feed sensor depth as a prior at training/eval time, getting the best of both SG-Free and sensor-aware regimes from one model.

**Evidence the swap is non-trivial.** The MapAnything paper reports that, with a *single* model, it matches or beats specialist feed-forward models across 12+ 3D tasks; in particular, on calibrated MVS and uncalibrated SfM benchmarks it is competitive with or beats VGGT (`papers/2509.13414v3.pdf` Tables 1–4). Replacing VGGT in a downstream pipeline with MapAnything is, in spirit, the same backbone-replacement that has worked many times in the foundation-model era (ResNet→ViT, CLIP→DINOv2, VGGT→VGGT-Det).

**Risk if wrong.** If the metric scale drifts more than ~10% on ARKit scenes, MGIoU on 8-corner reconstructions will be hurt (intersection terms become unstable). Mitigation in §8.

### 1.1 The two architectural upgrades

- **2D-anchored query generation (Boxer-style; `papers/2604.05212v1.pdf`).** Initialize 3D object queries from cheap 2D detections lifted via predicted depth, instead of purely FPS over a depth-projected point cloud.

  > **Intuition.** A learned query in 3DETR / VGGT-Det has to discover both *where to look* and *what's there*. Anchoring queries on 2D detections gives the decoder a category-aware spatial prior at layer 0 — the query already "knows" it's near a chair-shaped object. This is exactly the inductive bias that DETR3D / PETR / Boxer / WildDet3D all argue for.
  > **Evidence.** WildDet3D Table 5: removing the 2D head and predicting 3D directly **collapses AP from 30.2 → 11.1, a −19.1 AP drop**, with indoor sub-domains hit hardest (SUNRGBD: 33.9 → 5.1). This is the single largest ablation signal in the paper. Verified by `pdftotext` from `papers/2604.08626v2 (1).pdf`.
  > **Risk if wrong.** If the 2D detector misses an object, the 3D head never sees it. Mitigation: FPS fallback queries (§4.2.2).

- **MGIoU 3D loss (Le et al., AAAI 2026 Oral; `papers/2504.16443v2.pdf`).** Replace the axis-aligned GIoU loss in VGGT-Det with Marginalized GIoU, which is rotation-invariant, fully differentiable, and 10–40× faster than existing rotated-IoU losses (paper Abstract, verified via pdftotext).

  > **Intuition.** Indoor objects (chairs, tables, beds) are rarely axis-aligned; their yaw matters. Axis-aligned GIoU systematically over-rewards predictions that are rotated wrongly — the AABB envelope still overlaps. A rotation-aware loss is structurally necessary to get mAP@0.5 on rotated GT. MGIoU achieves this by computing 1D GIoU along the 6 candidate face-normal axes (`mgiou/losses.py:103`) and averaging — a separating-axis-theorem-style approximation that's smooth and cheap.
  > **Evidence.** Paper reports consistent gains over L1/GIoU/CIoU/rotated-IoU across rotated 2D detection, 3D detection, and quadrilateral regression (`papers/2504.16443v2.pdf` Tables 1, 3, 5). Implementation is pure PyTorch — no custom CUDA op (`mgiou/losses.py:8-12,74`).
  > **Risk if wrong.** At very small predicted box volumes (early training) the 1D projections all collapse and gradients vanish. Mitigation: L1 fallback for the first 2k steps and linear loss-weight warmup (already standard practice in the MGIoU2D reference implementation at `mgiou/losses.py:165-169`).

---

## 2. Goals, Non-Goals, and Success Criteria

**Goals.**

1. Design a 3D detection head that takes MapAnything's multi-view image features + predicted geometry and outputs 7-DoF oriented 3D boxes for indoor scenes.
2. Train **one unified closed-vocab model** jointly on Hypersim (synthetic dense) + WildDet3D-indoor (real in-the-wild), with ARKitScenes used **only for zero-shot generalization evaluation**.
3. Define 4 ablatable variants combining (MapAnything vs. VGGT) × (2D-anchor vs. FPS queries) × (MGIoU vs. axis-aligned GIoU) × (with/without DN + aleatoric), so each contribution is isolable.
4. Identify the top risks and mitigation strategies before writing any code.

**Non-goals.** Outdoor / LiDAR (KITTI, nuScenes); real-time deployment; end-to-end fine-tuning of MapAnything in iteration 1; **open-vocab / text-prompted detection** (closed-vocab only, by user decision — see §6.0); training *on* ARKitScenes.

**Why the joint training scope is the right one.**

> **Intuition.** Hypersim and WildDet3D are individually weak benchmarks for a foundation-model-backboned detector — Hypersim is too small (~46.6k images) to saturate a ViT-G-scale backbone, and WildDet3D's published baselines train on a much larger dataset mix that we can't match in compute. Jointly, they complement each other: Hypersim gives clean dense supervision (good for the loss landscape), WildDet3D gives diversity (good for generalization). The combination is *exactly* the regime where MapAnything's metric output should pay off — Hypersim grounds the metric scale, WildDet3D tests whether that grounding transfers.
> **Evidence.** WildDet3D's own published recipe (`WildDet3D/TRAINING_STRATEGY.md`) is a 3-stage multi-dataset curriculum mixing Omni3D + WildDet3D-Data + auxiliary datasets at carefully tuned ratios; they explicitly pitch this as enabling 6–10× faster training than single-dataset baselines (paper §4.3). Multi-dataset joint training is the SoTA recipe, not the experimental one.

**Success criteria** (reasonable bar for a paper-quality result):

| Metric | Target | Rationale |
|---|---|---|
| **Hypersim** mAP@0.25 (indoor val) | Establish baseline; report mAP@0.5 too | No published prior on this exact protocol; we set the reference. |
| **Hypersim** mAP@0.50 | ≥ 70% of Hypersim mAP@0.25 | Synthetic with clean GT — tight-IoU performance should be high. |
| **WildDet3D-Bench AP** (indoor-only subset, closed-vocab) | Within 5 AP of WildDet3D paper's *closed-vocab indoor* numbers at ≤ 30% of their training compute | Apples-to-apples isn't possible (different label space); the comparison is to their model evaluated on the same closed-vocab restriction. |
| **ARKitScenes zero-shot** mAP@0.25 | ≥ 50% of VGGT-Det's *trained-on-ARKit* number | A zero-shot model reaching half of in-domain performance is publishable. |
| **ARKitScenes zero-shot** mAP@0.50 | ≥ 30% of VGGT-Det's trained number | Same logic, tighter IoU. |
| Ablation clarity | Each of {backbone, query init, MGIoU, DN/aleatoric, joint-vs-single-dataset} contributes ≥ 0.5 mAP independently on Hypersim val | Required to attribute each component in the paper. |

> **Why "X% of in-domain" rather than absolute numbers for ARKit zero-shot.** ARKit's leaderboard depends heavily on training-set overlap; without training on ARKit, our absolute numbers will be lower by construction. The right comparison is "how close to in-domain does zero-shot get?" which is what published zero-shot 3D detection papers (3D-MOOD, DetAny3D, OV-3DET) also report.

---

## 3. Background

### 3.1 What VGGT-Det actually does (verified against source)

(All citations to `/Users/cuongpham/Documents/05_Projects/opensource/3DDET/VGGT-Det-CVPR2026/projects/VGGTDet/`.)

```
multi-view RGB (N=42 train / 81 test on ScanNet, 448×448)   ← config:162,199
        │
        ▼
   VGGT-1B (frozen, .eval())                                ← vggtdet.py:177-180
        │
        ├── aggregated_tokens_list (4 layers, 2048-dim)     ← vggtdet.py:302-333
        ├── depth_head    → per-view depth + confidence
        ├── camera_head   → extrinsics + intrinsics (OpenCV)
        └── attention maps → per-patch saliency
                 │
                 ▼
    unproject depth → 100K-point cloud in normalized scene coords
                 │
                 ▼
    attention-guided FPS                                    ← vggtdet.py:751-797
      priority = attention + 0.8 × distance                 ← config:102 (lambda_dist=0.8)
                 │
                 ▼
    Fourier coord PE (3D) → 512-dim query embeddings        ← config:15 (_token_dim_=512)
                 │
                 ▼
    8-layer DETR-style decoder, FFN dim 512                 ← config:16,51-55
      self-attn(queries) → cross-attn(queries, image tokens) → FFN
      return_intermediate=True (deep supervision)           ← vggtdet.py:820,824
                 │
                 ▼
    per-layer heads: cls + center (3D) + size (3D)
    n_reg_outs=6  (AXIS-ALIGNED, no yaw)                    ← config:63
                 │
                 ▼
    Hungarian matching: cls=1.0, center=0.0, obj=0.0, giou=2.0
                                                            ← config:68-73
```

Key choices we keep, modify, or drop:

| VGGT-Det choice | Disposition in MapDet3D | Why |
|---|---|---|
| Frozen backbone | **Keep** | Cuts training compute by ~6× (ViT-G is heavy); standard for foundation-model downstream tasks. |
| 8-layer decoder, 512-dim, FFN 512, 4 heads | **Keep as baseline** | Matches what trained successfully on ScanNet. Ablate to 6 / 8 / 12 layers later. |
| 256 queries | **Keep**, ablate 128 / 256 / 512 | ARKit has fewer GT boxes per scene than ScanNet; 256 may be over-provisioned. |
| Attention-guided FPS | **Compare** against 2D-anchor lifting (V1 vs V2) | Direct ablation. |
| Axis-aligned boxes (`n_reg_outs=6`) | **Drop.** Add 6D rotation (Gram-Schmidt) → 12 outputs. | Indoor objects have meaningful yaw; needed for mAP@0.5 on ARKit (which is OBB-annotated). |
| Axis-aligned GIoU | **Replace** with MGIoU 3D on oriented boxes | See §1.1 intuition. |
| Multi-layer VGGT features (4 layers) | **Adapt** — use MapAnything `[AAT-7, AAT-11, AAT-final]` plus encoder feat | Same philosophy: shallow features for localization, deep for semantics. |
| Per-layer deep supervision | **Keep** | Strong regularizer; verified in VGGT-Det via `return_intermediate=True`. |
| Per-scene point sample (100K) | **Keep** for the FPS-baseline variant only | Only needed when FPS is the query source. |
| Bicubic resize to 448 | **Change** to 518 | MapAnything's DINOv2 uses patch=14; 518=37×14 is the standard resolution. |

### 3.2 MapAnything outputs we will use (verified against source)

(All citations to `/Users/cuongpham/Documents/05_Projects/opensource/3DDET/map-anything/`.)

For each input view `i ∈ 1..N`, MapAnything returns a dict containing the following keys (subset; configurable via adaptors at `model.py:1780-1793`):

| Output key | Shape | Meaning | Use in MapDet3D |
|---|---|---|---|
| `pts3d` | (B, H, W, 3) | per-pixel 3D points, **metric, globally consistent frame** | Lift 2D detections → 3D query anchors; PE for memory tokens |
| `depth_along_ray` | (B, H, W, 1) | per-pixel depth along ray | Auxiliary; consistency loss vs. `pts3d` |
| `cam_quats` | (B, 4) | camera rotation (quaternion) | Compose with `pts3d` for view-aware PE |
| `cam_trans` | (B, 3) | camera translation (meters) | Same |
| `metric_scaling_factor` | (B,) | global scene scale | Verification only — we want pts3d already scaled |

> **Note on naming.** The *input* dict in `views[i]` uses the longer keys `camera_pose_quats`/`camera_pose_trans`; the *output* dict uses the shorter `cam_quats`/`cam_trans`. Source: `model.py:1523-1548` (inputs) vs `model.py:1780-1793` (outputs).

Intermediate features available for the decoder:

| Feature | Shape (per view) | Source |
|---|---|---|
| Encoder (DINOv2 ViT-G/14) | (h, w, 1536), where h=w=H/14=37 at 518² input | `model.py:1604-1624` |
| AAT intermediate layer 7 | (h, w, 1536) | `configs/model/info_sharing/...:12` (`intermediate_indices: [7, 11]`) |
| AAT intermediate layer 11 | (h, w, 1536) | same |
| AAT final (layer 15) | (h, w, 1536) | always returned |

That's **4 feature maps** per view at 37×37 resolution.

### 3.3 MGIoU 3D (verified against `mgiou/losses.py` and the paper PDF)

For two 3D OBBs represented as 8 corners `(B, 8, 3)`:

1. Compute three orthogonal face-normal axes for each box from corner edges `v0→v1`, `v0→v3`, `v0→v4` (`losses.py:49`). Concatenate the 6 axes (3 from pred + 3 from target) → shape `[6, 3]` (`losses.py:103`).
2. For each axis, project both boxes → 1D intervals; compute 1D GIoU (`losses.py:108-114`).
3. Average the 6 1D GIoU values → `mgiou ∈ [-1, 1]`. Loss = `(1 - mgiou) * 0.5 ∈ [0, 0.5]` (`losses.py:74`).

> **Intuition.** This is a soft separating-axis-theorem (SAT) approximation. SAT says two convex bodies are disjoint iff there exists a separating axis among their face normals. MGIoU averages the *signed* 1D overlap along all candidate normals: when the boxes are aligned and overlapping, every projection overlaps; when one is rotated wrong, several projections separate, dragging the average down. Unlike exact rotated 3D IoU (which requires polygon clipping in 3D and is non-differentiable at the polytope boundary), MGIoU is smooth everywhere.
> **Evidence.** Paper PDF confirms 10–40× speedup vs. exact rotated-IoU losses and metric-property + scale-invariance (`papers/2504.16443v2.pdf` Abstract). Class implementation verified at `mgiou/losses.py:27-119`. Pure PyTorch + `torch.vmap` (no CUDA op).
> **Risk if wrong.** Degenerate predicted volumes early in training. Mitigation in §8.

### 3.4 2D-anchored queries (verified against papers)

Two reference designs that both post-date VGGT-Det's submission and represent the current state of the art:

- **Boxer (`papers/2604.05212v1.pdf`).** Backbone: **DINOv3** (not DINOv2 — corrected). Takes off-the-shelf 2D box proposals (DETIC, OWLv2, SAM3), depth (optional), and posed images, lifts each 2D box to 3D via BoxerNet (transformer with self-attn → cross-attn → MLP head). Predicts **7-DoF OBB + aleatoric log σ²** uncertainty. Built on **Aria Gen1+Gen2, CA-1M, SUN-RGBD, ScanNet — explicitly excludes ARKitScenes due to CA-1M overlap** (corrected from v1). Released as **inference-only** code.
- **WildDet3D (`papers/2604.08626v2 (1).pdf`).** Box encoding (verbatim from paper §3.2):
  - `Δcx, Δcy` — center offset, normalized by scale factor `sc = 10`
  - `log d̂ = sd · log(d)`, `sd = 2.0` — log of metric depth
  - `log ŵ, ĥ, l̂ = sdim · log(w, h, l)`, `sdim = 2.0` — log of physical dimensions in meters
  - `r1..r6` — 6D rotation (first two rows of 3×3, Gram-Schmidt recovery)
  - **Total: 12 dims**
- The −19.1 AP ablation when removing the 2D head: see Table 5 of WildDet3D — "Removing the 2D head and predicting 3D boxes directly causes AP to collapse from 30.2 to 11.1 (−19.1), with indoor datasets hit hardest (SUNRGBD: 33.9 → 5.1, Objectron: 56.8 → 10.9)" (verified via pdftotext).

> **Why we adopt Boxer's 2D-anchor recipe but keep VGGT-Det's decoder shape.** Two reasons: (a) Boxer doesn't release training code, so we can't simply reuse its head; we need to re-implement the query-from-2D-box mechanism on top of a head we already understand and have working (VGGT-Det's). (b) Keeping the decoder constant across V0/V1/V2/V3 makes the ablations clean: only one knob changes per variant.

---

## 4. Proposed Architecture: MapDet3D

```
┌──────────────────────────────────────────────────────────────────────┐
│                  INPUT: N RGB views @ 518×518                        │
│              (N=8…16 indoor; N=2…4 wild-in-the-wild)                 │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              │
                              ▼
                  ┌──────────────────────┐
                  │   MapAnything        │
                  │   frozen, .eval()    │
                  │   DINOv2 ViT-G       │
                  │      ↓ (shared)      │
                  │   AAT 16-layer       │
                  └──────────────────────┘
                              │
        ┌─────────────────────┼──────────────────────┐
        ▼                     ▼                      ▼
 ┌──────────────────┐  ┌───────────────────┐  ┌──────────────────────┐
 │ 2D Detection Head│  │ Geometry outputs  │  │ (optional sensor     │
 │ DETR-style       │  │   pts3d (metric)  │  │  depth fed back IN   │
 │ shares DINOv2    │  │   depth_along_ray │  │  via depth_along_ray │
 │ feats; ~10M new  │  │   cam_quats/trans │  │  PRIOR input field)  │
 │ params; closed-  │  │   metric_scale    │  └──────────────────────┘
 │ vocab on unified │  └───────────────────┘
 │ ~60-class set    │
 │ JOINTLY trained  │
 │ with 3D head     │
 └──────────────────┘
        │                     │
        │       ┌─────────────┼──────────────────────────────┐
        │       ▼             ▼              ▼               ▼
        │  pts3d (metric) depth_ray   cam_quats/trans   metric_scale
        │       │
        │       └──┐
        │          ▼
        │   per-pixel 3D point cloud Pᵢ in world metric frame
        │          │
        │          ▼
        │   (optional fallback) attention-guided FPS → top-up queries
        ▼
 ┌────────────────────────────────────────────────────────────────┐
 │  Query Generator                                               │
 │                                                                │
 │  For each 2D detection (uᵢ,vᵢ) on view k with score sᵢ:        │
 │    1. p̂ᵢ = bilinear_sample(pts3d_k, uᵢ, vᵢ)  ∈ ℝ³ (metric)   │
 │    2. fᵢ = RoIAlign(0.5·enc_feat_k + 0.5·aat_final_k, bboxᵢ)   │
 │    3. eᵢ = Fourier-PE-3D(p̂ᵢ)  ∈ ℝ⁵¹²                           │
 │    4. qᵢ = MLP([proj_512(fᵢ); eᵢ; score_emb(sᵢ); cls_emb(cᵢ)]) │
 │                                                                │
 │  Top-K (default K=256) by 2D score; if N₂D < K, fill from FPS  │
 │  Optional: add denoising queries with noise levels (DN-DETR)  │
 └────────────────────────────────────────────────────────────────┘
                              │
                              ▼
 ┌────────────────────────────────────────────────────────────────┐
 │  Multi-Scale Image Memory                                      │
 │   • DINOv2 encoder feats (N, 37, 37, 1536) → proj → 512        │
 │   • AAT[7], AAT[11], AAT[15]   → proj → 512                    │
 │  Each token tagged with: view-id embedding +                   │
 │   3D-position PE built from pts3d at the token's pixel center  │
 │   (PETR-style: image tokens carry their own 3D coordinate)     │
 └────────────────────────────────────────────────────────────────┘
                              │
                              ▼
 ┌────────────────────────────────────────────────────────────────┐
 │  Decoder × 8 layers (DETR-style, pre-norm), FFN dim = 512      │
 │   For each layer:                                              │
 │     q = SelfAttn(q + PE3D(current box_center))                 │
 │     q = CrossAttn(q, memory + memory_PE)                       │
 │     q = FFN(q)                                                 │
 │   Per-layer aux head ⇒ deep supervision (matches VGGT-Det)     │
 └────────────────────────────────────────────────────────────────┘
                              │
                              ▼
 ┌────────────────────────────────────────────────────────────────┐
 │  Box & Class Heads (per query, per layer) — 12-dim + class     │
 │   • cls_logits ∈ ℝ^(C+1)                                       │
 │   • Δcenter ∈ ℝ³  (residual from anchor p̂ᵢ, in meters)         │
 │   • log_size ∈ ℝ³                                              │
 │   • rot6d ∈ ℝ⁶    (Gram-Schmidt → R ∈ SO(3))                   │
 │   • log_σ² ∈ ℝ⁷   (aleatoric, Boxer-style; optional in V3)     │
 └────────────────────────────────────────────────────────────────┘
                              │
                              ▼
 ┌────────────────────────────────────────────────────────────────┐
 │  Losses (Hungarian matching: cls + L1 center + MGIoU)          │
 │   • Cls: focal loss                                            │
 │   • Center: L1 in metric meters                                │
 │   • Size:   L1 on log_size                                     │
 │   • Rot:    chord loss on R (= ‖R_pred − R_gt‖_F)             │
 │   • Box:    MGIoU 3D on 8-corner reconstruction                │
 │   • (opt.) NLL with aleatoric σ²                               │
 │   • (opt.) Denoising loss on DN queries                        │
 └────────────────────────────────────────────────────────────────┘
```

### 4.1 Tensor shape spec (single training step)

Let `B` = batch (scenes), `N` = views per scene, `H=W=518`, patch=14 ⇒ `h=w=37`, `D_enc=1536`, `D_dec=512`, `K=256` queries, `C` = num classes.

| Tensor | Shape | Source |
|---|---|---|
| `images` | (B, N, 3, 518, 518) | dataloader |
| `pts3d` | (B, N, 518, 518, 3) | MapAnything, metric frame |
| `enc_feats` | (B, N, 37, 37, 1536) | MapAnything encoder |
| `aat_feats[i]` for i∈{7,11,15} | (B, N, 37, 37, 1536) | MapAnything AAT taps |
| `memory` (after proj+concat) | (B, N·37·37·4, 512) | multi-scale memory |
| `mem_pe_3d` | (B, N·37·37·4, 512) | PE3D(pts3d at each token's pixel) |
| `q_anchor_2d` | (B, K, 512) | from 2D detector + RoIAlign |
| `q_pe` | (B, K, 512) | PE3D(anchor p̂ᵢ) |
| `q_out[layer L]` | (B, K, 512) | per-layer decoder output |
| `cls_logits` | (B, L_dec, K, C+1) | per-layer head |
| `pred_center` | (B, L_dec, K, 3) | per-layer head |
| `pred_logsize` | (B, L_dec, K, 3) | per-layer head |
| `pred_rot6d` | (B, L_dec, K, 6) | per-layer head |
| `pred_corners` | (B, L_dec, K, 8, 3) | reconstructed for MGIoU |

### 4.2 Module-by-module specification

#### 4.2.1 2D detection head (closed-vocab, jointly trained, shared encoder)

> **Why we don't use OWLv2 / SAM3.** Our target is closed-vocab (§6.0). OWLv2 and SAM3 are open-vocab systems carrying a text encoder (CLIP-scale) and a large image encoder of their own. Under closed-vocab they are strictly *worse* than a focused closed-vocab detector: same or worse accuracy on the fixed taxonomy, several times the FLOPs, and a text-prompt pipeline we wouldn't use. The earlier v2 of this doc inherited "OWLv2 + SAM3" from Boxer's design, but Boxer is open-vocab — it has a different problem. Removing them is a strict improvement.

> **Choice (v3).** A small DETR-style 2D head, ~10M params, **reading MapAnything's DINOv2 ViT-G encoder features directly** (no separate backbone), **jointly trained** with the 3D head on the unified ~60-class closed-vocab taxonomy. Concretely: a 2D query set + 4-layer cross-attention decoder + classification + 2D-box-regression heads, sitting in parallel with the 3D-head decoder. The backbone is frozen; only the 2D head MLPs/attention layers are trainable.

> **Intuition (the central architectural argument).** Three reasons this is the right answer for our setup, in decreasing order of importance:
> 1. **We are *already* paying for DINOv2 ViT-G** because that's MapAnything's encoder. Adding a small head on top of features that already exist is essentially free in forward FLOPs (~3% extra) and adds no new backbone parameters. A separate detector (OWLv2 or RF-DETR as an external system) means running a second ViT — pure waste under closed-vocab.
> 2. **2D and 3D supervise each other when trained jointly.** WildDet3D's strongest published ablation (Table 5): removing the jointly-trained 2D head and predicting 3D directly drops AP from 30.2 → 11.1 (**−19.1 AP**). The mechanism is that the 2D-box loss is dense, low-noise supervision that anchors the 3D head's center/size regression in image space; it cannot be replicated by a *frozen external* detector because there's no gradient path. This is the published reason joint > frozen-external.
> 3. **Closed-vocab labels are free.** Hypersim provides per-pixel semantic+instance labels — 2D boxes are a `connected-components → bounding-box` away. WildDet3D provides 2D boxes natively (every 3D annotation has a corresponding 2D projection in the metadata). No new annotation, no separate 2D-detection pretraining stage.

> **Evidence.**
> - **3D-MOOD (arXiv 2507.23567, ICCV 2025).** "The first end-to-end 3D Monocular Open-set Object Detector, which lifts 2D detection into 3D space through a designed 3D bounding box head, enabling end-to-end joint training for both 2D and 3D tasks." Sets SOTA on Omni3D *including against closed-set methods*. This is the closest published precedent for our architecture choice; we adopt their joint-training pattern and replace their monocular backbone with MapAnything's multi-view encoder.
> - **WildDet3D paper Table 5** (verified verbatim by `pdftotext` against `papers/2604.08626v2 (1).pdf`): 2D head ablation = −19.1 AP. They use a jointly-trained head, not an external detector.
> - **RF-DETR (ICLR 2026, arXiv 2511.09554).** Shows that a small DETR head on top of a DINOv2 backbone is current SoTA for closed-vocab 2D detection on COCO. Our 2D head is essentially this architecture, sharing weights with MapAnything's encoder.
> - **Cube R-CNN (Omni3D, CVPR 2023).** The canonical closed-vocab indoor+outdoor monocular 3D detector — uses a 2D head + per-RoI 3D regressor. Same general philosophy.

> **Risk if wrong.**
> - *2D head competes for capacity with 3D head.* Mitigation: freeze backbone (already in plan); separate, non-shared decoders for 2D and 3D so they don't compete for cross-attention layers; balance loss weights with gradient-norm monitoring (`λ_2D` annealed to keep `||∇_2D|| / ||∇_3D||` ≈ 1.0).
> - *2D head doesn't converge on long-tail of unified taxonomy.* Mitigation: focal loss on 2D classification (standard for DETR); class-balanced sampling at the dataset level (already specified for the 3D loss).
> - *Joint training of 2D + 3D + multi-dataset is too many moving parts at once.* Mitigation: Stage-1 of curriculum (§7.2, Hypersim warm-up) explicitly trains both heads on a single clean dataset first; only S2 introduces WildDet3D.

> **What this changes for query generation.** §4.2.2 is unchanged except that the "2D detections" feeding the query generator come from our own jointly-trained 2D head, not from OWLv2. At test time we still use top-K 2D proposals to anchor 3D queries; at train time we can use either the 2D head's predictions OR the GT 2D boxes (mixed with a curriculum: 100% GT in S1 → 50/50 in S2 → 0% GT in S3) — analogous to teacher forcing in seq2seq training.

> **What we explicitly do NOT do.**
> - We do not fine-tune MapAnything's ViT-G backbone. Both heads read frozen features. (Possible follow-up: last 2-4 blocks at 0.1× LR, as WildDet3D does for their SAM3 encoder.)
> - We do not use external pre-trained closed-vocab detectors (RF-DETR / DEIMv2 / Cube R-CNN) as a frozen 2D source. Tried mentally: it costs an extra forward through a separate ViT, gives no joint-supervision signal, and adds a maintenance burden.

> **Architectural module spec.**
> - **Input:** `enc_feats` (B, N, 37, 37, 1536) from MapAnything's DINOv2 encoder.
> - **2D queries:** 100 learnable per view (so up to 100·N = 1600 candidate 2D boxes per scene at N=16). Smaller than the 3D-head's 256 — 2D recall on indoor scenes saturates well below 100.
> - **Decoder:** 4 layers, 256-dim (smaller than the 3D head's 512 — 2D is an easier task), 4 heads, pre-norm.
> - **Heads:** classifier (C+1 logits with focal loss) + 2D-box regressor (4 dims: cx, cy, w, h in normalized image coords, L1 + GIoU2D loss).
> - **Matching:** Hungarian per view, standard DETR cost (cls + L1 + GIoU2D).
> - **Output for 3D head:** top-K (K=K_3D=256) by 2D classification score across all views, with `(view_id, 2D_box, score, predicted_class)` → feeds query generator (§4.2.2 unchanged).

#### 4.2.2 Query Generator

```python
def generate_queries(pts3d, enc_feats, aat_final, det2d, K=256):
    """
    pts3d:     (B, N, H, W, 3)     metric world frame
    enc_feats: (B, N, h, w, D_enc) DINOv2 features
    aat_final: (B, N, h, w, D_enc)
    det2d:     list of (view_id, bbox_xyxy, score, class_prior) per scene
    Returns:   q (B,K,D_dec), q_pe (B,K,D_dec), anchor_p (B,K,3)
    """
    feats_for_roi = 0.5 * enc_feats + 0.5 * aat_final
    q_list = []
    for b in range(B):
        boxes = det2d[b]
        if len(boxes) < K:
            extra = att_guided_fps(pts3d[b], K - len(boxes))   # fallback
            boxes = boxes + extra
        boxes = top_k(boxes, K, by="score")
        for (vid, bb, sc, cp) in boxes:
            uv  = box_center(bb)
            p3d = sample_bilinear(pts3d[b, vid], uv)
            f   = roi_align(feats_for_roi[b, vid], bb)
            e   = fourier_pe_3d(p3d, dim=D_dec)
            q   = mlp([proj_512(f); e; score_emb(sc); cls_emb(cp)])
            q_list.append((q, e, p3d))
    return stack(q_list)
```

> **Intuition.** A query carries three priors: (1) a 3D position from MapAnything's pts3d at the 2D-box center, (2) a visual signature from RoIAligned features (semantics), (3) a class hint from the 2D detector. The decoder only needs to refine, not discover.
> **Evidence.** Identical structure to BoxerNet (paper §3.2). Fallback FPS matches 3DETR (`3detr/models/model_3detr.py:166-179`) and VGGT-Det (`vggtdet.py:751-797`).
> **Risk if wrong.** If 2D-box-center pixel falls on background (occluded object), the 3D anchor is wrong. **Mitigation:** sample 3 candidate pixels (box center + two random in-box pixels), keep the one with the most consistent depth (median of the three).

PE design: 3D Fourier features with 64 frequencies × 3 dims × {sin, cos} = 384, projected to 512. Same parameterization as `PositionEmbeddingCoordsSine` in 3DETR / VGGT-Det.

#### 4.2.3 Decoder

> **Choice.** 8 layers, 512-dim, 4 heads, FFN dim **512** (matches VGGT-Det exactly), pre-norm.
> **Intuition.** Use a known-good shape so the comparison vs. VGGT-Det is interpretable. Ablations on layer count come later.
> **Evidence.** VGGT-Det's config sets these exact values (`config/vggtdet_scannet.py:15,16,51-55`). The 512-FFN choice (rather than the more common 2048) is unusual but successful in their reported numbers; we inherit it deliberately.

Cross-attention key/value: multi-scale image memory with per-token 3D PE built from MapAnything's `pts3d` sampled at the token's pixel center. This is the PETR-style trick: image tokens carry their own 3D position so cross-attention is geometry-aware, not just appearance-aware. Self-attention uses the *updated* anchor-center PE (so the query "sees" its current 3D prediction at each layer).

#### 4.2.4 Box parameterization

> **Choice.** 12-dim output (Δcenter ∈ ℝ³, log_size ∈ ℝ³, rot6d ∈ ℝ⁶), matching WildDet3D's encoding (paper §3.2). Convert to 8 corners for MGIoU and for evaluation.
> **Intuition.** (a) Δcenter relative to anchor → small range → easier head learning. (b) log_size handles the wide dynamic range of indoor object sizes (TV remote → bed). (c) 6D rotation is the smallest representation guaranteed continuous over SO(3) — quaternions/Euler have discontinuities.
> **Evidence.** WildDet3D paper §3.2 verbatim; Zhou et al. 2019 "On the Continuity of Rotation Representations in Neural Networks" for the 6D rotation justification.
> **Risk if wrong.** A 4-fold symmetry ambiguity in yaw makes the rotation loss bimodal. **Mitigation:** canonical rotation normalization (W ≤ L; yaw folded to `[0, π)`) on GT before computing the loss — WildDet3D's exact recipe.

#### 4.2.5 Hypersim 9-DoF caveat

> **Issue.** Hypersim provides **9-DoF** boxes (3 position + full 3×3 rotation matrix + 3 scale), not 7-DoF OBBs. The 3×3 rotation may include shear/non-uniform-scale combinations that don't reduce to an OBB cleanly.
> **Decision.** At Hypersim load time, project the 9-DoF box onto its closest 7-DoF OBB approximation (SVD on the rotation×scale block to extract orthogonal R and diagonal S) and **discard** instances where the SVD residual > 5% (these are degenerate or thin objects unlikely to be useful for indoor detection benchmarks).
> **Evidence.** `apple/ml-hypersim` README explicitly defines 9-DoF; SVD projection to OBB is the standard fix used in OpenScene / EmbodiedScan preprocessing.

#### 4.2.6 Losses

Total loss combines the 2D head and the 3D head:

```
L_total = L_2D + L_3D

L_2D    = λ_2D_cls · FocalLoss(cls2d_logits, y_cls)        # 2D head only
        + λ_2D_l1  · L1(pred_2d_box, gt_2d_box)
        + λ_2D_giou · (1 − GIoU2D(pred_2d_box, gt_2d_box))

L_3D    = sum over decoder layers L of (
            λ_cls · FocalLoss(cls_logits, y_cls)
          + λ_ctr · L1(pred_center, gt_center)              # in meters
          + λ_sz  · L1(log_size, log(gt_size))
          + λ_rot · ChordLoss(R_pred, R_gt)                 # ‖R_p − R_g‖_F
          + λ_box · MGIoU3D(pred_corners, gt_corners)
          + λ_unc · NLL(pred, gt, σ²)                       # optional, V3
          + λ_dn  · DenoisingLoss(...)                      # optional, V3
        )
```

Defaults:
- **2D head:** `λ_2D_cls=2, λ_2D_l1=5, λ_2D_giou=2` (DETR conventions).
- **3D head:** `λ_cls=2, λ_ctr=5, λ_sz=1, λ_rot=2, λ_box=2, λ_unc=0.1, λ_dn=1`.
- 3D losses summed across 8 decoder layers with equal weight (deep supervision, matching VGGT-Det's `return_intermediate=True` setup at `vggtdet.py:820`). 2D loss has no deep supervision (only one 2D head output, after 4 decoder layers).

**Inter-head balancing.** The 2D and 3D losses can have very different gradient norms. We monitor `||∇_2D|| / ||∇_3D||` at the shared encoder feature (the only place gradients from both heads collide) and rescale `λ_2D_*` to keep the ratio in `[0.5, 2.0]`. If we hit the boundary, log a warning and step into the GradNorm algorithm (Chen et al. 2018) — simple, well-tested.

> **Why these weights.** `λ_ctr=5` mirrors VGGT-Det's center weighting. `λ_box=2` mirrors VGGT-Det's GIoU weighting (we just swap the loss function). `λ_cls=2` is a standard focal-loss weight in DETR derivatives. `λ_rot` is new (VGGT-Det was axis-aligned); we pick 2 to match `λ_box`'s order of magnitude and revisit if rotation loss dominates the gradient norm.

**Hungarian matching cost** (for assignment, not for backprop):
```
cost = 2·focal(cls) + 5·L1(center) + 2·MGIoU3D(corners) + 0.5·L1(log_size)
```
Optionally use *one-to-many* matching above an MGIoU threshold (default 0.1), matching VGGT-Det's `UnifiedMatcherMoreThanOne` mode and WildDet3D's auxiliary-head philosophy — gives denser positive supervision early in training.

---

## 5. Variants to Ablate

All variants are trained on **Hypersim + WildDet3D-indoor jointly** (see §7) unless noted, with identical optimizer / schedule / effective epochs. All are evaluated on Hypersim val, WildDet3D-Bench indoor val, and **zero-shot on ARKitScenes val** (no ARKit data ever seen at train time).

### 5.1 Component ablation ladder (the main paper table)

| Variant | Backbone | Query init | Box repr | Box loss | Extras | Purpose |
|---|---|---|---|---|---|---|
| **V0 (repro)** | VGGT-1B | Att-FPS | axis-aligned 6D | GIoU axis-aligned | — | Reproduce VGGT-Det numbers as control. |
| **V1 (swap)** | MapAnything | Att-FPS | OBB 12-dim | MGIoU 3D | — | Isolate backbone change. |
| **V2 (queries)** | MapAnything | 2D anchor | OBB 12-dim | MGIoU 3D | — | Isolate query mechanism. |
| **V3 (full)** | MapAnything | 2D anchor | OBB 12-dim | MGIoU 3D | DN + aleatoric | The proposed model. |
| **V2b (RGB-D)** | MapAnything + depth prior | 2D anchor | OBB 12-dim | MGIoU 3D | — | Quantifies RGB-D headroom on ARKit zero-shot eval (where sensor depth is available). |

> **Why this exact ladder.** V0 → V1 isolates the backbone change (the central claim of the doc). V1 → V2 isolates the query mechanism (the Boxer-style 2D anchoring, supported by WildDet3D's −19.1 AP ablation). V2 → V3 isolates training tricks. V2b tests whether MapAnything productively consumes sensor depth as a prior — only meaningful at *eval* time on ARKit since training data may or may not have depth.

### 5.2 Joint-training ablation (the secondary table)

To attribute the value of joint training, we additionally run V3 in three configurations:

| Sub-variant | Train data | Eval on Hypersim val | Eval on WildDet3D-indoor val | Eval zero-shot on ARKit val |
|---|---|---|---|---|
| **V3-H** | Hypersim only | ✓ in-domain | zero-shot | zero-shot |
| **V3-W** | WildDet3D-indoor only | zero-shot | ✓ in-domain | zero-shot |
| **V3-HW** | Hypersim + WildDet3D-indoor (joint) | ✓ joint-domain | ✓ joint-domain | zero-shot |

> **Intuition for what this measures.** V3-HW must beat both V3-H and V3-W on their respective in-domain val sets to claim joint training is *non-negative*. V3-HW's ARKit zero-shot AP, compared to V3-H and V3-W's ARKit numbers, tells us whether *diversity* (joint) or *cleanliness* (Hypersim-only) or *real-domain-realism* (WildDet3D-only) is the bigger driver of zero-shot generalization. WildDet3D's own paper Table 9 ablates dataset mixing in this style.
> **Risk if wrong.** If V3-HW < max(V3-H, V3-W) on either in-domain val, we have **negative transfer** — fix is dataset-conditioned BatchNorm-style normalization layers in the decoder (dataset-embedding adapter) or weighted gradient unscaling on the conflicting head.

---

## 6. Datasets, Taxonomy, and Splits (verified)

### 6.0 Unified closed-vocab taxonomy

We train on Hypersim + WildDet3D-indoor jointly, in a **closed-vocab** regime. The two datasets have incompatible label spaces (Hypersim ≈ NYU40-style ~40 indoor categories; WildDet3D ≈ 13,499 open-vocab classes), so the first design decision is the unified label space.

> **Choice.** A curated **~60-class indoor taxonomy**, defined as: NYU40 ∩ {Hypersim semantic classes} ∪ {top-20 most-frequent WildDet3D indoor classes not already in NYU40} ∪ `{"other"}`. Concretely: bed, chair, sofa, table, desk, dresser, bookshelf, bathtub, toilet, sink, refrigerator, oven, microwave, dishwasher, tv, monitor, lamp, pillow, curtain, door, window, … (full list to be finalized at preprocessing time).
> **Intuition.** NYU40 is the de-facto indoor 3D detection vocabulary (used by ScanNet, SUN RGB-D, ARKit). Hypersim semantic IDs already map cleanly to NYU40 (Hypersim uses NYU40 + a few extras). WildDet3D's most frequent indoor classes are a strict superset of NYU40 (the long tail is what makes WildDet3D special). A ~60-class union keeps the vast majority of training boxes from both datasets while remaining trainable with a single classifier head.
> **Evidence.** Hypersim README: "NYU40 + extra fine-grained labels". WildDet3D paper Fig. 6: 52% of WildDet3D-Data is indoor; the top-K indoor classes are dominated by furniture and appliances overlapping with NYU40. ARKit's 17 classes are a subset of NYU40 — making zero-shot eval clean (no held-out-class mapping needed beyond the "other" bucket).
> **Risk if wrong.** The "other" bucket dominates training (WildDet3D's long tail is huge). **Mitigation:** weight "other" loss down (λ=0.1) so it doesn't drown out the named classes; report per-class AP separately so the named-class signal is visible even if "other" AP is noisy.

> **Class mapping table.** Built at preprocessing time as `unified_taxonomy.json` with three columns: `unified_id, hypersim_nyu40_id, wilddet3d_category_names[]`. Generated by a deterministic script (no manual mapping needed for the well-known NYU40 part; manual review for the WildDet3D top-20 additions).

### 6.1 Training datasets

| Dataset | Type | #train (indoor only) | #val | #classes (orig.) | Box format | Notes |
|---|---|---|---|---|---|---|
| **Hypersim** | Synthetic V-Ray | **46,619 images** (74.6k public release after people/logo filtering) | 7,386 val | NYU40 + extras | **9-DoF** (3 pos + 3×3 R + 3 scale) — reduce to 7-DoF OBB (§4.2.5) | Scene-level split. Documented minor asset reuse across train/test. ~10 boxes/image. |
| **WildDet3D-indoor** | RGB in-the-wild (real) | **~53,549 images** (52% × 102,979 human-verified) | ~1,284 (52% × 2,470) | 13,499 (we keep top-60 indoor + "other") | 12-dim (Δcx/Δcy/10, log d × 2, log w,h,l × 2, r1..r6) | We use **human-verified split only**, skip the 896k synthetic split for compute reasons. |

> **Why "human only" for WildDet3D.** The 896k synthetic split is generated by SAM3 + FoundationPose; using it would mean we're indirectly training on SAM3's biases. The 102,979 human-verified images give us cleaner supervision and a much faster training loop. WildDet3D's own ablation (paper Table 8) shows that human-only training is within 2 AP of human+synthetic — a worthwhile tradeoff for 9× lower compute.
> **Evidence.** WildDet3D paper §B.1 dataset proportions; their Stage-1 trains on Omni3D only (no synthetic) and reaches competitive numbers.

### 6.2 Joint training mix and stages

Following WildDet3D's verified 3-stage recipe (`WildDet3D/TRAINING_STRATEGY.md`, paper Table 9), adapted to our 2-dataset setup:

| Stage | Data mix | Epochs | LR schedule | Init |
|---|---|---|---|---|
| **S1: Hypersim warm-up** | 100% Hypersim | 8 | warmup 1k steps + cosine | Scratch (frozen MapAnything; decoder + heads init) |
| **S2: Joint training** | 50% Hypersim + 50% WildDet3D-indoor (uniform per-image sampling within each split) | 16 | warmup 500 steps + cosine | S1 checkpoint |
| **S3: Real-data emphasis** | 25% Hypersim + 75% WildDet3D-indoor | 6 | LR × 0.1, no warmup | S2 checkpoint |

**Total: 30 epochs.** Comparable to VGGT-Det's 400 ScanNet epochs in wall-clock terms because (a) we have ~2× more training data per epoch and (b) we use ViT-G (heavier forward). WildDet3D paper §4.3 claims 27 epochs total, so this is well-precedented.

> **Why this exact curriculum.** S1 establishes a clean loss landscape on dense synthetic data — MGIoU 3D and the rotation head get a stable initialization before real-data noise enters. S2 is the main joint training stage where both datasets contribute equally. S3 emphasizes real data because the eventual deployment target is real (ARKit zero-shot, and real downstream scenes). This mirrors WildDet3D's Stage 2 → Stage 3 transition (their Stage 3 also emphasizes their real-data subset).
> **Evidence.** WildDet3D `TRAINING_STRATEGY.md` §2 (3-stage curriculum verified vs. their code).
> **Risk if wrong.** Hypersim's "domain weight" in S2 may be too high (synthetic dominates gradients). **Mitigation:** if Hypersim val AP keeps improving but WildDet3D-indoor val AP stalls during S2, drop Hypersim share to 25% earlier.

**Sampling.** Per-step, sample a *scene* from each dataset proportionally (50/50 in S2), then sample N views from each scene. Use a `DatasetMixer` that interleaves at the batch level (not within a batch) to avoid mixing label spaces in a single Hungarian matching call.

### 6.3 Zero-shot evaluation on ARKitScenes

> **Choice.** Evaluate the joint-trained model on ARKit val *without any fine-tuning*. Report mAP@0.25 and mAP@0.50 on all 17 ARKit classes.
> **Class mapping.** All 17 ARKit classes (bed, chair, sofa, table, refrigerator, oven, microwave, dishwasher, washer, tv_monitor, bathtub, toilet, sink, cabinet, shelf, fireplace, stairs) are a subset of NYU40 and a strict subset of our unified taxonomy. The mapping is 1:1 with no "other" fallback needed.
> **Evaluation protocol.** ARKit's standard split (5.6k val frames). Use VGGT-Det's evaluation code to ensure leaderboard comparability.
> **Intuition.** This is the central generalization claim of the paper. A model trained on synthetic + diverse-real, evaluated zero-shot on a third real domain (LiDAR iPad), measures whether MapAnything's geometric priors plus our detection head actually transfer.
> **Evidence.** Closely matches the zero-shot protocol used by 3D-MOOD, DetAny3D, and OV-3DET (all train on Omni3D / synthetic, eval zero-shot on ARKit).
> **Caveat.** We do *not* expect to beat VGGT-Det's in-domain ARKit number (it trains on ARKit). Our target (§2) is to reach ≥50% of VGGT-Det's number — a publishable bar for zero-shot 3D detection.

---

## 7. Training Recipe

The high-level curriculum (S1 / S2 / S3) is in §6.2. Below is the per-stage detail.

### 7.1 Hyperparameters (apply across all stages unless overridden)

| Hyperparameter | Value | Why |
|---|---|---|
| Optimizer | AdamW, β=(0.9, 0.999), wd=1e-4 | VGGT-Det `config:279-280`; standard for DETR derivatives. |
| Base LR | **1e-4** decoder, **0** backbone (frozen) | Lower than VGGT-Det's 2.5e-4 because we have a larger backbone and a 2-dataset mix — more noisy gradients warrant a smaller step. WildDet3D paper uses 1e-4 (`TRAINING_STRATEGY.md` §1). |
| Schedule | warmup → cosine within each stage, η_min=1e-6 | Warmup of 1k steps for S1, 500 for S2, none for S3. |
| Batch | 2 scenes × 16 views × 8 GPUs (eff. 16 scenes) | Memory-constrained on H100; ViT-G + AAT is heavy. |
| Precision | bfloat16 | VGGT-Det uses bf16. WildDet3D promotes LayerNorms back to FP32 for bf16 stability (`wilddet3d/model.py:325-332`) — we adopt this. |
| Grad clip | 35.0 in S1/S2, **0.1** in S3 (matches WildDet3D's S3 stability fix) | The S3 LR drop + small grad clip is the recipe WildDet3D found stable on real-data emphasis. |
| Input resolution | 518 × 518 | DINOv2 patch=14; 37×37 token grid. |
| Views per scene | **8 Hypersim, 4 WildDet3D** | Hypersim is multi-view per scene; WildDet3D is mostly single-view in-the-wild, so we sample N=4 views from the spatially-closest neighbors in the same scene (when scene metadata exists) or duplicate the single image with augmentation. |
| Queries K | 256 (ablate 128 / 256 / 512) | Matches VGGT-Det's 256 (`config:90`). |
| Decoder layers | 8 (ablate 6 / 8 / 12) | Matches VGGT-Det. |
| Augmentations | photometric (color jitter, gamma); view-drop p=0.1; random origin shift σ=0.7 m; per-dataset color matching | The per-dataset color matching shifts WildDet3D's mean image stats toward Hypersim's during S2, narrowing the domain gap before features reach the frozen DINOv2 backbone. |

### 7.2 Per-stage detail

| | **S1: Hypersim warm-up** | **S2: Joint** | **S3: Real emphasis** |
|---|---|---|---|
| Data | 100% Hypersim | 50/50 Hypersim/WildDet3D-indoor | 25/75 Hypersim/WildDet3D-indoor |
| Epochs | 8 | 16 | 6 |
| LR | 1e-4 | 1e-4 | 1e-5 |
| Warmup | 1k steps | 500 steps | — |
| Grad clip | 35.0 | 35.0 | 0.1 |
| MGIoU weight λ_box | 0 → 2 over first 2k steps, then 2 | 2 | 2 |
| Loss notes | L1-only on box for first 2k steps (MGIoU warmup, §8) | All losses active | Same as S2 |

**Hardware budget.** MapAnything ViT-G ≈ 1.8B params (frozen). Trainable: decoder + heads ≈ 60M. Single step at bf16, batch 2 scenes × 16 views × 518² ≈ **55 GB** → fits on 80 GB H100; falls back to 12 views on 40 GB A100. Total wall-clock estimate at 8×H100: S1 ≈ 12h, S2 ≈ 36h, S3 ≈ 9h, total **~57h (~2.5 days)** per variant.

> **Caching strategy.** As in v2: pre-compute per-image **single-view encoder features** to disk (frozen, deterministic). AAT cross-view features must be re-computed because they depend on the view bag chosen for each batch. Saves ~50% of forward time. Cache budget for indoor splits: 46.6k (Hypersim) + 53.5k (WildDet3D-indoor) ≈ 100k images × 37×37×1536×2 B (bf16) = **~400 GB** on local NVMe. Manageable.

---

## 8. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| MapAnything's metric scale drifts across scenes (>5–10%) | M | H | Verify on Hypersim (known GT scale) before training; if drift > 5%, learn a per-scene rescale token in the decoder. |
| 2D detector misses small objects (TVs, stoves) ⇒ no query ⇒ no detection | H | M | Fallback FPS queries (specified). Lower score threshold + more candidates (K_2D = 512 → top-256 selected by decoder). |
| MGIoU on degenerate predictions early in training is flat | M | M | L1 fallback for first 2k steps (already standard in `MGIoU2D` at `losses.py:165-169`); warmup λ_box from 0 → 2 linearly over 5k steps. |
| MapAnything ViT-G + 16-layer AAT ~4× slower per forward than VGGT-1B | H | M | Cache **single-view encoder features** per scene (frozen, deterministic). Cross-view AAT must be recomputed when view bag changes. Saves ~50% epoch time, not 70% (corrected from v1). |
| Hypersim 9-DoF → 7-DoF conversion loses signal | M | M | SVD residual filter (§4.2.5); also keep a "9-DoF Hypersim" variant later if filtering throws too many boxes. |
| Hypersim ⇒ ARKit domain gap larger than expected | M | M | Mix Hypersim:ARKit 1:1 in fine-tuning rather than sequential transfer; add light color matching aug. |
| 6D rotation + canonical normalization mis-handled at eval ⇒ silent AP drop | M | H | Unit-test the W≤L normalization on synthetic boxes before any training run; visualize first 100 GT boxes per dataset and confirm canonical form. |
| Hungarian matching collapses (queries assigned to a few classes) | L | H | Use one-to-many matching from epoch 1 (VGGT-Det's `UnifiedMatcherMoreThanOne`); add DN queries in V3. |
| "Why not just fine-tune the backbone?" reviewer Q | H | L | Run a small backbone-FT ablation (1 variant, 30 epochs); document. |
| **Negative transfer between Hypersim and WildDet3D** (joint training hurts both) | M | H | Watch V3-HW vs. V3-H/V3-W during S2. If either in-domain val regresses by >1 mAP from S1, add dataset-conditional layer-norm scales (DA-conditional adapter, ~50k extra params, well-precedented in multi-dataset detection). |
| **Unified taxonomy mismatch — Hypersim's NYU40 doesn't cover a WildDet3D class** that turns out to be common at eval | M | M | Add the missing class to `unified_taxonomy.json` and rerun S3 only (cheap re-finetune); keep "other" as the safety net. |
| **WildDet3D-indoor 52% split is hand-wavy** — paper Fig. 6 reports indoor share but doesn't release a per-image scene-type label | M | M | Use a simple text-classifier on WildDet3D's scene metadata strings (provided per image) to filter; fall back to keeping all images if filter drops >40%. |
| **ARKit zero-shot AP is low** (we'd hoped for ≥50% of in-domain, get ≤30%) | M | M | Report what we get; add a "ARKit oracle 2D" variant where ARKit's GT 2D boxes replace the trained 2D head's predictions — separates query-recall failures from 3D-head failures. |
| **2D head's gradient swamps 3D head's** (or vice versa) | M | M | GradNorm-style auto-rebalancing of `λ_2D_*` (§4.2.6); fall back to two-stage training (freeze 2D head after S1, train only 3D head in S2–S3) if instability persists. |

---

## 9. Implementation Plan (not in this doc)

1. **Week 1.** Stand up MapAnything inference-only; verify pts3d metric scale on Hypersim (which has GT mesh + GT scale) for 20 scenes; write `MapAnythingFeatureExtractor` wrapper with disk cache.
2. **Week 2.** Build `unified_taxonomy.json` (Hypersim NYU40 + top-20 WildDet3D indoor). Build Hypersim 9-DoF → 7-DoF SVD-projection preprocessing pipeline. Build WildDet3D-indoor split filter.
3. **Week 3.** Port VGGT-Det's decoder + matcher into a new `MapDet3D` repo, parameterized by `(backbone, query_init, box_loss, extras)`. Train V0 (VGGT control) on Hypersim-only as a sanity check (we won't have a published baseline to compare, but it lets us validate the pipeline).
4. **Week 4.** Build the **jointly-trained 2D head** (4-layer DETR-style on shared DINOv2 encoder features). Wire 2D-head output → query generator. Train V1 jointly (2D + 3D) on Hypersim + WildDet3D-indoor (the 3-stage curriculum). At this step also extract 2D box GT from Hypersim's instance segmentation (connected components → bbox).
5. **Week 5.** Add MGIoU 3D loss + 6D rotation + canonical normalization; train V2 on the same joint mix.
6. **Week 6.** Train V3 (DN + aleatoric). Run V3-H and V3-W single-dataset ablation variants in parallel.
7. **Week 7.** ARKit zero-shot evaluation on all variants. V2b (RGB-D depth-prior) eval on ARKit.
8. **Week 8.** Ablations, final runs, paper draft.

---

## 10. Open Questions (require empirical resolution)

1. **Metric scale stability across datasets.** Does MapAnything output pts3d at consistent metric scale across Hypersim (synthetic, no real LiDAR) vs. WildDet3D (real, varied focal lengths)? **Test:** overlay pts3d on Hypersim GT meshes (20 scenes) and on the small subset of WildDet3D images that have LiDAR depth GT.
2. **Unified taxonomy final size.** Is 60 classes the right number? **Test:** plot per-class box-count histogram for both datasets after mapping; if the head of the distribution is concentrated in 20-30 classes, drop the long tail to keep the loss focused.
3. **Joint-training balance.** Is 50/50 the right Hypersim/WildDet3D ratio in S2? **Test:** sweep {25/75, 50/50, 75/25} on a 4-epoch S2 run; pick the ratio with the best joint val mAP.
4. **GT-2D teacher-forcing schedule.** The 2D head's predictions feed the query generator during training. Should we use 100% GT 2D boxes early in training and anneal to 100% predicted? **Test:** sweep schedules `{100→0%, 100→50%, 50→0%, always-mixed-50%}` on a short S2 run. The right schedule balances "3D head sees clean queries early" vs. "3D head sees realistic 2D-head noise at test time".
5. **Variable N at train vs. test time.** MapAnything is robust by design, but is the decoder? **Test:** N=4/8/16/24 at test time with fixed N=8 train.
6. **AAT taps:** all three `{7, 11, 15}` vs. last only? VGGT-Det's ablation found multi-layer > last-only; verify on MapAnything.
7. **Per-dataset BN-style adapter for the decoder.** If §8 Risk "negative transfer" materializes, would a dataset-conditional layer-norm scale (~50k extra params) fix it? **Test:** add when needed; measure delta on V3-HW.
8. **ARKit zero-shot ceiling.** What fraction of in-domain ARKit performance can a zero-shot model realistically reach? Published 3D-MOOD / DetAny3D numbers say roughly 30-50% — we aim for 50%; if we hit 70%, that's a result worth investigating.

---

## 11. References

| # | Paper / Repo | Citation in this doc | Local copy |
|---|---|---|---|
| 1 | Keetha et al., *MapAnything*, 3DV 2026, [arXiv:2509.13414](https://arxiv.org/abs/2509.13414), [code](https://github.com/facebookresearch/map-anything) | Backbone | `papers/2509.13414v3.pdf`, `map-anything/` |
| 2 | Xu et al., *VGGT-Det*, CVPR 2026, [arXiv:2603.00912](https://arxiv.org/abs/2603.00912) | Architectural template | `papers/2603.00912v1.pdf`, `VGGT-Det-CVPR2026/` |
| 3 | Le et al., *MGIoU*, AAAI 2026 Oral, [arXiv:2504.16443](https://arxiv.org/abs/2504.16443), [project](https://ldtho.github.io/MGIoU/) | Loss | `papers/2504.16443v2.pdf`, `MGIoU/` |
| 4 | DeTone et al., *Boxer*, Meta Reality Labs, [arXiv:2604.05212](https://arxiv.org/abs/2604.05212) | Query-from-2D recipe | `papers/2604.05212v1.pdf`, `boxer/` |
| 5 | Huang et al., *WildDet3D*, [arXiv:2604.08626](https://arxiv.org/abs/2604.08626) | Dataset + 2D-anchoring evidence | `papers/2604.08626v2 (1).pdf`, `WildDet3D/` |
| 6 | Misra et al., *3DETR*, ICCV 2021 | Decoder template, query PE | `3detr/` |
| 7 | Roberts et al., *Hypersim*, ICCV 2021 | Pretraining dataset | — |
| 8 | Baruch et al., *ARKitScenes*, NeurIPS 2021 | Primary benchmark | — |
| 9 | Zhou et al., *On the Continuity of Rotation Representations in Neural Networks*, CVPR 2019 | 6D rotation justification | — |
| 10 | DETR3D / PETR / DN-DETR | 2D-to-3D PE, denoising training | — |
| 11 | Yang et al., *3D-MOOD: Lifting 2D to 3D for Monocular Open-Set Object Detection*, [arXiv:2507.23567](https://arxiv.org/abs/2507.23567), ICCV 2025 | Direct precedent for joint 2D+3D end-to-end training | — |
| 12 | Robinson et al., *RF-DETR: Neural Architecture Search for Real-Time Detection Transformers*, [arXiv:2511.09554](https://arxiv.org/abs/2511.09554), ICLR 2026 | DINOv2-backboned closed-vocab 2D detection SoTA — informs our 2D-head design | — |
| 13 | Chen et al., *GradNorm: Gradient Normalization for Adaptive Loss Balancing*, ICML 2018 | 2D/3D loss auto-balancing | — |
| 14 | Brazil et al., *Omni3D / Cube R-CNN*, CVPR 2023 | Canonical closed-vocab indoor+outdoor 3D detector precedent | — |

---

## Appendix A. Symbol glossary

`B` batch; `N` views/scene; `H,W` image (518); `h,w` token grid (37); `D_enc` 1536; `D_dec` 512; `K` queries (256); `C` classes; `OBB` oriented bounding box; `MGIoU` Marginalized Generalized IoU; `AAT` Alternating-Attention Transformer; `IFR` Intermediate Feature Return; `DN` Denoising queries.

## Appendix B. Pseudocode — single training step

```python
# 1. Backbone (frozen)
with torch.no_grad():
    views = [{"img": images[:, i], "data_norm_type": "dinov2"} for i in range(N)]
    map_out = map_anything(views)        # list of N dicts
    pts3d   = stack([v["pts3d"] for v in map_out], dim=1)            # (B,N,H,W,3)
    enc_f   = stack([v["enc_feat"] for v in map_out], dim=1)         # (B,N,h,w,D_enc)
    aat_f   = stack([v["aat_feats"] for v in map_out], dim=1)        # (B,N,3,h,w,D_enc)

# 2. Trainable 2D head (shares enc_f, no separate backbone)
det2d_pred = head_2d(enc_f)                                          # list of (vid, bbox, score, cls)
loss_2d    = compute_2d_loss(det2d_pred, gt_2d_boxes)                # focal + L1 + GIoU2D

# 3. Build queries from 2D predictions (mix GT during training per schedule)
det2d_for_queries = mix_pred_and_gt(det2d_pred, gt_2d_boxes, gt_ratio=schedule(step))
q, q_pe, anchor_p = generate_queries(pts3d, enc_f, aat_f[..., -1, :], det2d_for_queries, K=256)

# 4. Memory
mem, mem_pe = build_memory(enc_f, aat_f, pts3d)                      # (B, N*h*w*4, 512)

# 5. 3D Decoder
q_outs = decoder(q, mem, q_pe, mem_pe)                               # 8-list of (B,K,512)

# 6. 3D Heads
preds = [box_head(q_out, anchor_p) for q_out in q_outs]              # cls, Δc, log_s, R, σ²

# 7. Canonical-form normalization on GT
gt_canon = canonical_obb(gt)                                         # W≤L, yaw ∈ [0,π)

# 8. Matching + 3D loss
losses_3d = []
for layer_pred in preds:
    matches = hungarian(layer_pred, gt_canon, cost_fn=cost_with_mgiou)
    losses_3d.append(compute_loss(layer_pred, gt_canon, matches,
                                  use_mgiou=True, λ_box=anneal(step)))

# 9. Combine 2D + 3D loss with auto-balancing
loss_3d = sum(losses_3d) / len(losses_3d)
total   = grad_norm_balance(loss_2d, loss_3d, shared_param=enc_f)
total.backward()
```

---

*End of design document v2.*

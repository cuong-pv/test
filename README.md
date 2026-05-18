# MapDet3D-Mono-DepthPC-Last

**Single-image RGB-D 3D object detection. Point cloud from sensor depth + intrinsics, features from MapAnything's last AAT layer only.**

MapDet3D-Mono-DepthPC-Last predicts 7-DoF oriented 3D bounding boxes for indoor furniture and appliances given **one RGB image, its aligned metric depth map, and camera intrinsics**. The 3D point cloud used for query lifting and positional encoding comes from **unprojecting the input depth map with intrinsics K** — a deterministic geometric step. The 3D decoder's cross-attention memory is built from **a single feature map** — MapAnything's last Alternating-Attention Transformer layer (AAT layer 15) at 37×37 resolution.

This variant is the lightest of the MapDet3D family: deterministic geometry from a calibrated sensor + a single-layer feature memory that trains faster and uses less GPU memory. It is the recommended production target on platforms with calibrated metric depth and known intrinsics.

---

## At a glance

| | |
|---|---|
| **Input** | One RGB image + one aligned metric depth map + **3×3 camera intrinsics `K` (required)** |
| **Output** | List of 7-DoF 3D OBBs `(center_xyz, size_whl, R, class, score)`, camera frame, meters |
| **Point cloud source** | **Sensor depth unprojected with `K`** — no learned geometric output is used |
| **Backbone** | MapAnything, frozen. Uses only AAT layer 15 as the cross-attention memory source; encoder features are used for the 2D head. All other backbone outputs (predicted pts3d, depth, camera pose, intermediate AAT layers) are computed but discarded. |
| **Memory tokens** | 1369 (= 37 × 37 × 1 level) |
| **Detection head** | DETR-style with 2D-anchored 3D queries, MGIoU 3D loss, joint 2D + 3D training |
| **Vocabulary** | Closed-set, **30 predefined indoor classes** (`C = 30`) |
| **Train data** | Hypersim + WildDet3D-indoor-with-depth (joint, 3-stage curriculum) |
| **Eval data** | Hypersim val, WildDet3D-indoor-with-depth val (in-domain) + ARKitScenes val (zero-shot) |
| **License** | Apache 2.0 (matches MapAnything's permissive checkpoint) |
| **Status** | Architecture spec + training guide; reference implementation pending |

---

## Why use this model

Four properties make it deployment-friendly:

1. **The depth sensor sets the metric scale.** The 3D point cloud is unprojected directly from sensor depth + intrinsics — a deterministic geometric operation with no learned parameters and no scale ambiguity. Industrial / clinical / robotic deployments with calibrated metric depth get more accurate 3D points than a learned model would predict.
2. **Predictable failure modes.** Unprojected depth fails in obvious, debuggable ways (depth == 0 → no 3D point). The model uses a 3-tier snap-to-valid fallback to handle depth holes gracefully without anchoring queries at the origin.
3. **Lower memory + faster cross-attention.** The 3D decoder uses a single-layer memory (1369 tokens) instead of fusing four feature taps (5476 tokens) — about ~25% of the cross-attention FLOPs, ~30% lower peak training memory (~38 GB vs. ~55 GB per GPU at batch 2 × 16), ~10–15% end-to-end inference speedup.
4. **Most of MapAnything's complexity is unused at inference.** Deployment-time forward pass can skip MapAnything's geometric prediction heads (config flag `backbone_geometry_heads=False`), saving an additional ~5–10% backbone compute.

This model is **not** a real-time edge model. The DINOv2 ViT-G backbone is ~1.8B parameters frozen; expect ~130 ms / frame on an H100, ~350 ms on an A100, ~1.3 s on an RTX 4090 at 518×518 input.

**When this variant is appropriate.** Your deployment scenario has calibrated metric depth from a hardware sensor (iPad LiDAR, Realsense, Kinect, ZED, or similar) and known camera intrinsics, and you want the lightest possible runtime.

**When it is not appropriate.** Your depth source is sparse, very noisy, or unreliable (consumer phones in challenging lighting), or you need to operate from RGB only with no intrinsics. For those cases the sibling `MapDet3D-Mono` (multi-scale memory + learned pts3d) is the right choice — it tolerates missing intrinsics and uses MapAnything's learned hole-filling.

---

## Quickstart (planned inference API)

```python
from mapdet3d_mono_depthpc_last import MapDet3DMonoDepthPCLast
import numpy as np
from PIL import Image

model = MapDet3DMonoDepthPCLast.from_pretrained("mapdet3d-mono-depthpc-last-v1").cuda().eval()

rgb   = np.array(Image.open("room.jpg"))                   # (H, W, 3), uint8
depth = np.load("room_depth.npy").astype(np.float32)       # (H, W), meters
K     = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])    # REQUIRED

boxes = model.predict(
    rgb=rgb,
    depth=depth,
    intrinsics=K,                  # REQUIRED — no fallback in this variant
    score_threshold=0.3,
)

# Returns a list of dicts; see docs/ARCHITECTURE.md §2.4 for the schema.
```

---

## Repo layout

```
MapDet3D-Mono-DepthPC-Last/
├── README.md                          ← this file
├── docs/
│   ├── ARCHITECTURE.md                ← self-contained design spec
│   └── TRAINING.md                    ← dataset prep, curriculum, hyperparameters
├── mapdet3d_mono_depthpc_last/        ← (pending) reference implementation
│   ├── model.py
│   ├── inference.py
│   └── configs/
└── checkpoints/                       ← (pending) pretrained weights
```

---

## Documentation

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — full design spec. Self-contained. Includes input/output contract (intrinsics required), the depth-unprojection module, valid-mask plumbing, snap-to-valid query fallback, single-layer feature memory, 3D decoder, prediction heads, losses, and evidence/intuition for every design choice.
- **[docs/TRAINING.md](docs/TRAINING.md)** — how to reproduce or fine-tune. Dataset preparation (incl. Hypersim depth ray→axis conversion and WildDet3D depth-availability filter), the 3-stage joint curriculum, hyperparameters, evaluation including zero-shot on ARKitScenes.

---

## Hardware requirements

| Stage | Min GPU | Recommended GPU | Notes |
|---|---|---|---|
| **Inference** | RTX 3090 (24 GB) | H100 (80 GB), A100 (40+ GB) | bf16; 518×518 input; one image |
| **Fine-tuning** | A100 40 GB × 1 | H100 80 GB × 8 | bf16; staged curriculum; ~38 GB peak per GPU |
| **Full training from scratch** | H100 80 GB × 4 | H100 80 GB × 8 | ~45 hours end-to-end |

See [docs/TRAINING.md](docs/TRAINING.md) for the breakdown.

---

## Sibling variants

Three other variants exist with the same detection head but different architectural choices for memory and pts3d source. Choose based on deployment constraints:

| Variant | Memory | pts3d source | Inputs required | Best for |
|---|---|---|---|---|
| `MapDet3D-Mono` | Multi-scale (5476 tokens) | MapAnything predicted | RGB + depth | RGB-D, K unknown |
| `MapDet3D-Mono-DepthPC` | Multi-scale (5476 tokens) | Depth + K unprojection | RGB + depth + K | Calibrated, multi-scale-cautious |
| `MapDet3D-Mono-Last` | Single layer (1369 tokens) | MapAnything predicted | RGB + depth | RGB-D, want speed |
| **`MapDet3D-Mono-DepthPC-Last`** (this) | Single layer (1369 tokens) | Depth + K unprojection | RGB + depth + K | **Calibrated, want speed (production target)** |

Weights are not interchangeable across variants — each was trained against a different `pts3d` distribution and/or memory configuration. Pick the variant matching your deployment and use its pretrained checkpoint.

---

## Citation

If you use MapDet3D-Mono-DepthPC-Last in research, please cite the upstream works it builds on:

```
@article{keetha2026mapanything,
  title   = {MapAnything: Universal Feed-Forward Metric 3D Reconstruction},
  author  = {Keetha, Nikhil and others},
  journal = {3DV},
  year    = {2026},
  url     = {https://arxiv.org/abs/2509.13414}
}
@article{le2026mgiou,
  title   = {Marginalized Generalized IoU: A Unified Objective Function for Optimizing Any Convex Parametric Shapes},
  author  = {Le, Duy-Tho and Pham, Trung and Cai, Jianfei and Rezatofighi, Hamid},
  journal = {AAAI},
  year    = {2026},
  url     = {https://arxiv.org/abs/2504.16443}
}
@article{xu2026vggtdet,
  title   = {VGGT-Det: Mining VGGT Internal Priors for Sensor-Geometry-Free Multi-View Indoor 3D Object Detection},
  author  = {Xu, Dan and others},
  journal = {CVPR},
  year    = {2026},
  url     = {https://arxiv.org/abs/2603.00912}
}
@article{yang2025_3dmood,
  title   = {3D-MOOD: Lifting 2D to 3D for Monocular Open-Set Object Detection},
  journal = {ICCV},
  year    = {2025},
  url     = {https://arxiv.org/abs/2507.23567}
}
```

---

## License

Apache 2.0, matching the permissive MapAnything checkpoint license. Training data terms (Hypersim, WildDet3D, ARKitScenes) apply separately — see each dataset's license before redistributing weights trained on them.

---

## Contact

Open an issue on the repository, or contact Cuong Pham (phamcuong92.hust@gmail.com).

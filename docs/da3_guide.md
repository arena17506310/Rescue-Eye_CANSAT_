# DA3 (Depth Anything 3) 최적 활용 가이드
# RTX 3080 10GB 환경 기준

> 이 문서는 `analysis/pipeline/reconstruct.py`, `depth.py`, `mesh_export.py` 작업 시 참조.
> DA3 공식 README 기반으로 RESCUE-EYE 파이프라인에 맞게 정리.

---

## 1. DA3의 역할: 메인 3D 복원 엔진

DA3는 RESCUE-EYE 파이프라인의 **핵심 엔진**이다. 기존 COLMAP → DA3 순차 구조에서
**DA3 단독으로 포즈 추정 + Dense Depth + 3D 메시를 모두 처리**하는 구조로 변경되었다.

```
[기존] COLMAP(포즈) → DA3(depth만)
[현재] DA3(포즈+depth+3D 메시) → COLMAP은 fallback 전용
```

### DA3가 수행하는 작업

| 기능 | 설명 | 출력 |
|------|------|------|
| 카메라 포즈 추정 | `use_ray_pose=True`로 자체 추정 | `prediction.extrinsics` [N,3,4] |
| Dense Depth | 멀티뷰 일관된 깊이 맵 | `prediction.depth` [N,H,W] |
| 신뢰도 맵 | 픽셀별 추정 신뢰도 | `prediction.conf` [N,H,W] |
| 카메라 내부 파라미터 | 초점거리 등 자동 추정 | `prediction.intrinsics` [N,3,3] |
| 3D 메시 export | glb/ply 직접 출력 | export_dir에 파일 저장 |

---

## 2. 모델 선택: DA3-LARGE-1.1

| 모델 | 파라미터 | 예상 VRAM | RTX 3080 적합성 |
|------|---------|----------|----------------|
| DA3-GIANT-1.1 | 1.15B | ~8GB | 단독 사용 시 가능하나, 파이프라인 내에서 OOM 위험 |
| **DA3-LARGE-1.1** | **0.35B** | **~4GB** | **최적 — 파이프라인 VRAM 예산(~4GB) 정확히 충족** |
| DA3-BASE | 0.12B | ~2GB | 가능하나 정확도 손실 |
| DA3-SMALL | 0.08B | ~1.5GB | 정확도 부족 |

- `-1.1` 접미사 모델은 학습 버그가 수정된 리프레시 버전이다. **반드시 `-1.1`을 사용할 것.**
- DA3-LARGE는 VGGT와 동등한 성능을 보인다 (공식 README 명시).
- Nested 시리즈(DA3NESTED-GIANT-LARGE)는 1.40B 파라미터로 10GB VRAM에서 불가.

### HuggingFace 모델 ID

```
depth-anything/DA3-LARGE-1.1
```

---

## 3. 설치

```bash
conda activate rescue-eye

# 의존성 (setup.md에서 PyTorch + CUDA 설치 완료 전제)
pip install xformers

# DA3 설치
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3
pip install -e .

# 3D Gaussian Splatting export 사용 시 (gs_ply, gs_video 포맷)
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf
pip install -e ".[all]"
```

### 설치 검증

```python
from depth_anything_3.api import DepthAnything3
import torch
model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE-1.1")
model = model.to(device=torch.device("cuda"))
print("DA3-LARGE-1.1 로드 성공")
print(f"VRAM 사용: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
```

---

## 4. 파이프라인 내 DA3 사용법

### 4.1 메인 경로: DA3 단독 추론

DA3가 **포즈 추정 + depth + 3D 메시를 한 번에 처리**한다. COLMAP 불필요.

```python
from depth_anything_3.api import DepthAnything3

model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE-1.1")
model = model.to(device="cuda")

# DA3 단독: 포즈 + depth + 3D 메시 한 번에 생성
prediction = model.inference(
    images,                    # 이미지 경로 리스트 또는 PIL/numpy
    use_ray_pose=True,         # 자체 포즈 추정 (더 정확, 약간 느림)
    export_dir="output",
    export_format="glb"        # glb, ply, npz, gs_ply, gs_video
)

# 결과
prediction.depth          # [N, H, W] float32 — depth maps
prediction.conf           # [N, H, W] float32 — confidence maps (핵심: fallback 판정 기준)
prediction.extrinsics     # [N, 3, 4] float32 — camera poses (w2c, OpenCV 컨벤션)
prediction.intrinsics     # [N, 3, 3] float32 — camera intrinsics
# + export_dir에 glb 파일 자동 생성
```

### 4.2 COLMAP Fallback 경로

DA3 신뢰도(`prediction.conf`)가 낮으면 COLMAP으로 포즈만 복원 후 DA3에 재주입한다.

```python
def run_reconstruction(images: list[str]) -> Prediction:
    """DA3 메인 → 신뢰도 낮으면 COLMAP fallback"""
    prediction = da3_model.inference(
        images, use_ray_pose=True,
        export_dir="output", export_format="glb"
    )

    if prediction.conf.mean() >= DA3_CONFIDENCE_THRESHOLD:  # 0.5
        logger.info("DA3 단독 성공")
        return prediction

    # Fallback: COLMAP 포즈 → DA3 재추론
    logger.warning("DA3 신뢰도 낮음 → COLMAP fallback")
    colmap_poses = run_colmap_sfm(images)
    prediction = da3_model.inference(
        images, poses=colmap_poses,
        export_dir="output", export_format="glb"
    )
    return prediction
```

### 4.3 use_ray_pose 옵션

| 설정 | 속도 | 정확도 | 용도 |
|------|------|--------|------|
| `use_ray_pose=True` (RESCUE-EYE 기본) | 느림 | 향상 | 메인 경로 — 이미지 수 적을 때 최적 |
| `use_ray_pose=False` | 빠름 | 기본 | 이미지 20장+ 대량 처리 시 |

낙하 촬영 특성상 이미지 수가 제한적이므로 `use_ray_pose=True`가 기본값이다.

---

## 5. Export 포맷 가이드

| 포맷 | 용도 | RESCUE-EYE 활용 |
|------|------|----------------|
| **glb** | 3D 메시 | **기본 — 심사 시연 + 웹 뷰어용** (`mesh_export.py`) |
| **ply** | Point Cloud | Voxel 분석 입력용 (`voxel/converter.py`) |
| **npz** | depth/conf/pose 배열 | 후처리 + 디버깅용 |
| gs_ply | 3D Gaussian Splatting | novel view synthesis (선택) |
| gs_video | GS 렌더링 비디오 | 시연 영상 (선택, gsplat 필요) |

파이프라인에서는 **glb**(시각화) + **ply/npz**(분석)를 동시에 내보내는 것을 권장.

---

## 6. VRAM 최적화 전략

### 6.1 파이프라인 내 VRAM 예산

```
순차 실행 (한 번에 하나만 GPU에 로드):
  DA3-Large  ~4GB  → 해제    ← 메인 (포즈+depth+3D)
  SAM2       ~2GB  → 해제
  Florence-2 ~2GB  → 해제
  ─────────────────────────
  피크: ~4GB (DA3 단독 실행 시)

  [fallback 시에만]
  COLMAP     ~1GB  → 해제    ← DA3 신뢰도 낮을 때만
  DA3-Large  ~4GB  → 해제    ← COLMAP 포즈로 재추론
```

**모델 간 전환 시 반드시 이전 모델 해제:**

```python
del model
torch.cuda.empty_cache()
```

### 6.2 배치 크기 관리

```python
# shared/config.py
DA3_BATCH_SIZE = 4    # RTX 3080 10GB 기준 안전한 최대값
```

- 이미지 해상도가 높으면(1920x1080+) 배치를 **2**로 줄일 것
- OOM 발생 시 배치를 절반으로 줄이고 재시도

```python
def run_da3_safe(model, images, batch_size=4):
    """OOM 안전 DA3 추론 — 배치 자동 축소"""
    while batch_size >= 1:
        try:
            results = []
            for i in range(0, len(images), batch_size):
                batch_imgs = images[i:i+batch_size]
                pred = model.inference(
                    batch_imgs,
                    use_ray_pose=DA3_USE_RAY_POSE,
                    export_dir="output",
                    export_format=DA3_EXPORT_FORMAT,
                )
                results.append(pred)
            return results
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch_size //= 2
            logger.warning(f"OOM — 배치 축소: {batch_size}")
    return None
```

### 6.3 처리 해상도 조절

DA3 CLI에는 `--process-res-method` 옵션이 있다.

- 캔위성 카메라(Global Shutter) 원본 해상도가 높을 경우, VRAM 절약을 위해 리사이즈 고려
- 권장: 긴 변 기준 최대 768px (VRAM 절약 + 충분한 품질)

---

## 7. CLI 활용 (디버깅 및 단독 테스트용)

### 7.1 Backend 모드 (모델 상주)

모델을 GPU에 캐시하여 반복 추론 시 로딩 시간을 절약한다.

```bash
export MODEL_DIR=depth-anything/DA3-LARGE-1.1
export GALLERY_DIR=workspace/da3_output
mkdir -p $GALLERY_DIR

# 백엔드 시작 (모델을 GPU에 상주)
da3 backend --model-dir ${MODEL_DIR} --gallery-dir ${GALLERY_DIR}

# 이미지 폴더 추론 (포즈+depth+3D 메시 한 번에)
da3 auto /path/to/selected_images \
    --export-format glb \
    --export-dir ${GALLERY_DIR}/test_run \
    --use-backend
```

### 7.2 단독 실행 (1회성)

```bash
da3 auto /path/to/selected_images \
    --export-format glb \
    --export-dir ${GALLERY_DIR}/test_run \
    --model-dir ${MODEL_DIR}
```

---

## 8. DA3 출력 → Voxel 파이프라인 연결

DA3 출력물을 Voxel 변환 엔진(`analysis/voxel/`)에 전달하는 흐름:

```python
# DA3 출력 (메인 경로든 fallback이든 동일한 구조)
depth = prediction.depth          # [N, H, W] 미터 단위
conf = prediction.conf            # [N, H, W] 신뢰도
extrinsics = prediction.extrinsics  # [N, 3, 4] 카메라 포즈
intrinsics = prediction.intrinsics  # [N, 3, 3] 내부 파라미터

# depth + pose → 3D Point Cloud 생성
# 각 픽셀을 3D 공간으로 역투영 (back-projection)
for i in range(N):
    K = intrinsics[i]           # [3, 3]
    T = extrinsics[i]           # [3, 4] w2c
    d = depth[i]                # [H, W]
    c = conf[i]                 # [H, W]

    # 신뢰도 낮은 픽셀 제거
    mask = c > DA3_CONFIDENCE_THRESHOLD

    # 픽셀 좌표 → 카메라 좌표 → 월드 좌표
    # points_3d = unproject(K, T, d, mask)
    # → analysis/voxel/converter.py 에서 Voxel Grid 변환

# 최종: Point Cloud → Voxel Grid → Void 탐지 → 부피 계산
# + glb 메시는 별도로 mesh_export.py에서 시각화/심사용으로 관리
```

---

## 9. 주의사항 및 트러블슈팅

### DA3 포즈 추정 불안정

- **원인**: 이미지 overlap 부족, 급격한 고도 변화, 모션 블러
- **대응**: COLMAP fallback이 자동 발동됨 (`conf.mean() < 0.5`)
- **예방**: 촬영 간격 줄이기, Blur 임계값 올리기

### DA3 confidence 전부 낮음

- **원인**: 이미지 품질 자체가 낮음 (전부 블러 등)
- **대응**: `DA3_CONFIDENCE_THRESHOLD`를 0.5 → 0.3으로 낮추기
- **주의**: 임계값을 너무 낮추면 부정확한 3D 복원이 통과될 수 있음

### glb export 실패

- **원인**: gsplat 미설치 (gs_ply/gs_video 사용 시)
- **해결**: `pip install --no-build-isolation git+...gsplat.git`
- **참고**: glb/ply/npz는 gsplat 없이도 동작

### XFormers 관련

RTX 3080(Ampere, SM 86)은 XFormers를 정상 지원한다. 문제 발생 시:
```bash
pip install xformers --force-reinstall
```

### OOM 발생 시 체크리스트

1. 다른 모델이 GPU에 남아있지 않은지 확인 (`nvidia-smi`)
2. `torch.cuda.empty_cache()` 호출 확인
3. `DA3_BATCH_SIZE`를 4 → 2 → 1로 축소
4. 입력 이미지 해상도 축소 (긴 변 768px 이하)
5. `use_ray_pose=False`로 전환 (VRAM 소폭 절약)
6. 그래도 실패 시 DA3-BASE(0.12B, ~2GB)로 모델 다운그레이드

### Metric Depth vs Relative Depth

- `DA3-LARGE-1.1`은 **relative depth**를 출력한다.
- DA3 자체 포즈 추정 모드에서 스케일 보정이 포함됨.
- 정밀 metric depth가 필요하면 `DA3METRIC-LARGE`(0.35B, Apache 2.0)를 별도 사용 가능.
- Voxel 부피 오차가 큰 경우 DA3 metric depth로 Point Cloud 스케일 재보정 고려.

### 라이선스

- `DA3-LARGE-1.1`: **CC BY-NC 4.0** (비상업적 사용만 허용)
- 대회 및 학술 목적 사용은 문제없음
- `DA3-BASE`, `DA3-SMALL`, `DA3METRIC-LARGE`: Apache 2.0 (상업적 사용 가능)

---

## 10. 설정값 요약 (shared/config.py 반영)

```python
# DA3 설정 (메인 3D 복원 엔진)
DA3_MODEL_ID              = "depth-anything/DA3-LARGE-1.1"  # -1.1 필수
DA3_MODEL_SIZE            = "large"
DA3_BATCH_SIZE            = 4              # OOM 시 2 → 1로 축소
DA3_USE_RAY_POSE          = True           # 자체 포즈 추정 (기본 ON)
DA3_EXPORT_FORMAT         = "glb"          # glb, ply, npz, gs_ply, gs_video
DA3_CONFIDENCE_THRESHOLD  = 0.5            # 이하 시 COLMAP fallback 발동
MAX_INFERENCE_RES         = 768            # 긴 변 최대 해상도

# COLMAP (fallback 전용)
COLMAP_GPU_INDEX          = 0
```

---

*RESCUE-EYE DA3 Guide v2.0 — RTX 3080 10GB 최적화 (DA3 메인 + COLMAP fallback)*

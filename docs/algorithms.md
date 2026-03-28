# RESCUE-EYE 핵심 분석 알고리즘
# 이 문서는 analysis/ 작업 시 참조

---

## 1. 분석 파이프라인 전체 흐름 (DA3 메인 + COLMAP fallback)

```
SD카드 이미지들
    │
    ▼ [전처리]
    Blur 수치 로그 기반 이미지 선별
    │
    ▼ [1단계] DA3 단독 추론 (메인 경로)
    use_ray_pose=True로 자체 포즈 추정 + Dense Depth + 3D 메시(glb)
    출력: depth map, confidence map, camera poses, Point Cloud, 3D 메시
    │
    ├─ confidence >= THRESHOLD → 메인 경로 계속
    │
    └─ confidence < THRESHOLD → [COLMAP fallback]
       │  COLMAP SfM으로 포즈만 복원
       │  복원된 포즈를 DA3에 입력하여 depth + 3D 재생성
       │
    ▼ [2단계] SAM2
    Void 후보 주변 구조물 인스턴스 분리
    → Lean-to / V-shape / Pancake 유형 판별 + 안정성 점수
    │
    ▼ [3단계] Florence-2
    도메인 프롬프트로 시각적 입구 탐지
    Region 단위 정밀 이해 → 낮은 오탐률
    │
    ▼ [융합] Fusion Logic
    confirmed / estimated / low_priority 등급 판정
    │
    ▼ [Voxel 분석]
    Point Cloud → Voxel Grid → Void 클러스터
    → 부피(m³) 계산 → 1.5m³ 기준 생존 판정
    │
    ▼ [출력]
    3D 구조 우선순위 지도 + 3D 메시(glb) 뷰어
    확정(빨강) / 추정(주황) / 주의(노랑)
    각 지점: 예상 부피 + 내부 깊이 + 최적 진입 경로
```

### DA3 핵심 API 사용법

```python
from depth_anything_3.api import DepthAnything3

model = DepthAnything3.from_pretrained("depth-anything/da3-large")
model = model.to(device="cuda")

# 메인 경로: DA3 단독 (포즈 + depth + 3D 메시)
prediction = model.inference(
    images,                    # 이미지 경로 리스트 또는 PIL/numpy
    use_ray_pose=True,         # 자체 포즈 추정 (더 정확, 약간 느림)
    export_dir="output",
    export_format="glb"        # glb, ply, npz, gs_ply, gs_video
)

# 결과 접근
prediction.depth          # [N, H, W] float32 — depth maps
prediction.conf           # [N, H, W] float32 — confidence maps
prediction.extrinsics     # [N, 3, 4] float32 — camera poses (w2c)
prediction.intrinsics     # [N, 3, 3] float32 — camera intrinsics

# COLMAP fallback 경로: DA3에 외부 포즈 주입
prediction = model.inference(
    images,
    poses=colmap_poses,        # COLMAP에서 복원한 포즈
    export_dir="output",
    export_format="glb"
)
```

### DA3 export 포맷 가이드

| 포맷 | 용도 |
|------|------|
| glb | 3D 메시 — 웹 뷰어/심사 시연용 (권장) |
| ply | Point Cloud — Voxel 분석 입력용 |
| npz | depth + pose 수치 데이터 — 후처리용 |
| gs_ply | 3D Gaussian Splatting — novel view synthesis |
| gs_video | Gaussian Splatting 비디오 — 시연 영상용 |

### Fallback 판정 로직

```python
def run_reconstruction(images: list[str]) -> Prediction:
    """DA3 메인 → 신뢰도 낮으면 COLMAP fallback"""
    prediction = da3_model.inference(
        images, use_ray_pose=True,
        export_dir="output", export_format="glb"
    )
    
    if prediction.conf.mean() >= DA3_CONFIDENCE_THRESHOLD:
        logger.info("DA3 단독 성공")
        return prediction
    
    logger.warning("DA3 신뢰도 낮음 → COLMAP fallback")
    colmap_poses = run_colmap_sfm(images)
    prediction = da3_model.inference(
        images, poses=colmap_poses,
        export_dir="output", export_format="glb"
    )
    return prediction
```

---

## 2. Voxel 변환 파이프라인 (`analysis/voxel/`)

```
Point Cloud (N×3, float32)
    │  voxel_size = 0.05m (5cm 격자)
    ▼
Occupied Voxel Grid (bool 3D array)
    │  6-connectivity Flood Fill — 바깥 경계에서 시작
    ▼
External Voxels (외부 공기와 연결된 빈 공간)
    │  NOT(Occupied) AND NOT(External)
    ▼
Void Voxels (내부 고립 공간 = 생존 후보)
    │  Connected Component Labeling
    ▼
Void Clusters
    │  volume_m3 = voxel_count × (0.05)³
    ▼
생존 판정: volume_m3 >= 1.5  →  후보 목록 등록
```

---

## 3. 4모델 융합 판정 로직 (`analysis/pipeline/fuse.py`)

```python
def classify_void(void: VoidCandidate, florence, da3, sam2) -> str:
    """
    confirmed : 시각적 입구 + 물리적 깊이 + 구조적 지지 모두 확인
    estimated : 입구 가려졌지만 3D 공간 존재 + 구조적 지지 확인
    low_priority : 위 조건 미충족
    """
    has_visual_entry   = florence.detected                       # Florence-2
    has_physical_depth = da3.max_depth_m > 0.5                   # DA3
    has_structure      = sam2.has_lean_to or sam2.has_v_shape    # SAM2
    meets_volume       = void.volume_m3 >= 1.5                   # 최소 생존 부피

    if has_visual_entry and has_physical_depth and has_structure:
        return "confirmed"
    elif meets_volume and has_structure:
        return "estimated"
    else:
        return "low_priority"
```

---

## 4. 이미지 선별 로직 (`analysis/preprocess/selector.py`)

```python
def select_images(image_dir: str, blur_log: dict[str, float]) -> list[str]:
    """
    blur_log: {filename: blur_score}
    BLUR_THRESHOLD 이상(선명한 이미지)만 반환.
    """
    selected = [
        os.path.join(image_dir, fname)
        for fname, score in blur_log.items()
        if score >= BLUR_THRESHOLD and os.path.exists(os.path.join(image_dir, fname))
    ]
    logger.info(f"선별: {len(selected)} / {len(blur_log)} 장 사용")
    return selected
```

---

## 5. 핵심 데이터 모델 (`shared/data_models.py`)

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class ImageQuality:
    filename: str
    blur_score: float          # Laplacian 분산값 (높을수록 선명)
    timestamp_ms: int          # 촬영 시각 (낙하 시작 기준 ms)
    gimbal_pitch_deg: float    # 촬영 시 카메라 각도

@dataclass
class VoidCandidate:
    id: str
    structure_type: str        # "lean_to" | "v_shape" | "pancake" | "pocket"
    volume_m3: float
    confidence: str            # "confirmed" | "estimated" | "low_priority"
    priority_score: float      # 0.0 ~ 1.0
    center_xyz: tuple          # 3D 좌표 (m)
    entry_point: Optional[tuple]
    stability_score: float     # 0.0(위험) ~ 1.0(안전)
    access_difficulty: float   # 0.0(쉬움) ~ 1.0(어려움)

@dataclass
class AnalysisResult:
    void_candidates: list      # List[VoidCandidate]
    point_cloud: np.ndarray    # (N, 3) float32
    voxel_grid: np.ndarray     # (X, Y, Z) bool
    voxel_size_m: float        # 0.05
    images_used: int
    processing_time_sec: float
```

---

## 6. 전역 설정값 상세 (`shared/config.py`)

```python
# 하드코딩 절대 금지. 모든 수치는 이 파일에서만 관리.
import os

# 캔위성 촬영
CAPTURE_INTERVAL_SEC = 0.5
GIMBAL_TARGET_PITCH  = -65.0   # 지면 방향 (°)
BLUR_THRESHOLD       = 100.0   # Laplacian 분산 임계값

# 통신
BT_BAUD_RATE  = 115200         # 주의: 공장 기본값 9600 → 반드시 변경
GPS_BAUD_RATE = 9600
IMU_BAUD_RATE = 115200

# Voxel 분석
VOXEL_SIZE_M       = 0.05
VOID_MIN_VOLUME_M3 = 1.5       # 성인 1명 생존 최소 부피
DEPTH_MIN_M        = 0.5

# DA3 설정 (메인 3D 복원 엔진)
DA3_MODEL_SIZE    = "large"        # "large" 권장 (~4GB) / "giant"은 ~8GB로 OOM 위험
DA3_BATCH_SIZE    = 4              # 배치 초과 시 OOM 발생
DA3_USE_RAY_POSE  = True           # 자체 포즈 추정 (더 정확, 약간 느림)
DA3_EXPORT_FORMAT = "glb"          # glb, ply, npz, gs_ply, gs_video
DA3_CONFIDENCE_THRESHOLD = 0.5     # 이하 시 COLMAP fallback 발동

# COLMAP (fallback 전용)
COLMAP_GPU_INDEX  = 0              # GPU 특징점 매칭용

# SAM2 / Florence-2
SAM2_MODEL_TYPE   = "sam2_l"       # SAM2 Large
```

---

## 7. AI 추론 에러 처리 패턴

```python
def run_da3(images: list[str]) -> Optional[Prediction]:
    """DA3 메인 추론 + COLMAP fallback + OOM 처리"""
    try:
        prediction = da3_model.inference(
            images, use_ray_pose=DA3_USE_RAY_POSE,
            export_dir="output", export_format=DA3_EXPORT_FORMAT
        )
        
        if prediction.conf.mean() >= DA3_CONFIDENCE_THRESHOLD:
            logger.info(f"DA3 단독 성공: depth shape={prediction.depth.shape}")
            return prediction
        
        # Fallback: COLMAP 포즈 → DA3 재추론
        logger.warning("DA3 신뢰도 낮음 → COLMAP fallback")
        colmap_poses = run_colmap_sfm(images)
        prediction = da3_model.inference(
            images, poses=colmap_poses,
            export_dir="output", export_format=DA3_EXPORT_FORMAT
        )
        return prediction
        
    except torch.cuda.OutOfMemoryError:
        logger.error("VRAM 부족 — DA3_BATCH_SIZE 줄이기")
        torch.cuda.empty_cache()
        return None
    except Exception as e:
        logger.error(f"DA3 실패: {e}", exc_info=True)
        return None
```

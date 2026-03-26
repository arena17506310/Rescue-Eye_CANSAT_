# RESCUE-EYE 핵심 분석 알고리즘
# 이 문서는 analysis/ 작업 시 참조

---

## 1. 분석 파이프라인 전체 흐름

```
SD카드 이미지들
    │
    ▼ [전처리]
    Blur 수치 로그 기반 이미지 선별
    │
    ▼ [1단계] COLMAP SfM
    카메라 포즈 복원 (항공 이미지 특화, GPS 고도로 초기화)
    출력: 각 이미지의 위치/방향 + Sparse Point Cloud
    │
    ▼ [2단계] Depth Anything 3 (DA3)
    COLMAP 포즈를 입력으로 멀티뷰 Dense Depth 생성
    실제 물리 거리(m) 직접 예측 + 스케일 보정 완료
    출력: Dense Point Cloud (m 단위 확정)
    ※ DUSt3R + Depth Anything V2 역할을 DA3 단일 모델로 통합
    │
    ▼ [3단계] SAM2
    Void 후보 주변 구조물 인스턴스 분리
    → Lean-to / V-shape / Pancake 유형 판별 + 안정성 점수
    │
    ▼ [4단계] Florence-2
    도메인 프롬프트로 시각적 입구 탐지 (Grounding DINO 대체)
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
    3D 구조 우선순위 지도
    확정(빨강) / 추정(주황) / 주의(노랑)
    각 지점: 예상 부피 + 내부 깊이 + 최적 진입 경로
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

# AI 모델 (VRAM 관리)
COLMAP_GPU_INDEX  = 0              # COLMAP GPU 특징점 매칭용
DA3_MODEL_SIZE    = "large"        # "large" 권장 (~4GB) / "giant"은 ~8GB로 OOM 위험
SAM2_MODEL_TYPE   = "sam2_l"       # SAM2 Large
DA3_BATCH_SIZE    = 4              # 배치 초과 시 OOM 발생
```

---

## 7. AI 추론 에러 처리 패턴

```python
def run_da3(images: list[np.ndarray], poses: np.ndarray) -> Optional[np.ndarray]:
    try:
        result = da3_model.inference(images)
        logger.info(f"DA3 완료: depth shape={result.depth.shape}")
        return result
    except torch.cuda.OutOfMemoryError:
        logger.error("VRAM 부족 — DA3_BATCH_SIZE 줄이기")
        torch.cuda.empty_cache()
        return None
    except Exception as e:
        logger.error(f"DA3 실패: {e}", exc_info=True)
        return None
```

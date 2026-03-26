# 하드코딩 절대 금지. 모든 수치는 이 파일에서만 관리.
import os

# ── 캔위성 촬영 ────────────────────────────────────────────
CAPTURE_INTERVAL_SEC: float = 0.5
GIMBAL_TARGET_PITCH: float  = -65.0   # 지면 방향 (°), 음수 = 아래
BLUR_THRESHOLD: float       = 100.0   # Laplacian 분산 임계값 (낮추려면 50)

# ── 통신 ──────────────────────────────────────────────────
BT_BAUD_RATE:  int = 115200   # 주의: 공장 기본값 9600 → 반드시 변경
GPS_BAUD_RATE: int = 9600
IMU_BAUD_RATE: int = 115200

# ── Voxel 분석 ────────────────────────────────────────────
VOXEL_SIZE_M:       float = 0.05   # 5cm 격자
VOID_MIN_VOLUME_M3: float = 1.5    # 성인 1명 생존 최소 부피 (m³)
DEPTH_MIN_M:        float = 0.5    # DA3 유효 깊이 하한 (m)

# ── AI 모델 (VRAM 관리 — RTX 3080 10GB) ──────────────────
COLMAP_GPU_INDEX: int = 0              # COLMAP GPU 특징점 매칭용
DA3_MODEL_SIZE:   str = "large"        # "large" 권장 (~4GB) / "giant"은 ~8GB로 OOM 위험
SAM2_MODEL_TYPE:  str = "sam2_l"       # SAM2 Large
DA3_BATCH_SIZE:   int = 4              # 배치 초과 시 OOM 발생

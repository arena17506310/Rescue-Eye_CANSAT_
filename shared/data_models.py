from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class ImageQuality:
    """낙하 중 촬영된 이미지 1장의 품질 메타데이터.

    Args:
        filename:        SD카드 내 파일명
        blur_score:      Laplacian 분산값 (높을수록 선명)
        timestamp_ms:    촬영 시각 (낙하 시작 기준 ms)
        gimbal_pitch_deg: 촬영 시 카메라 각도 (°)
    """
    filename: str
    blur_score: float
    timestamp_ms: int
    gimbal_pitch_deg: float


@dataclass
class VoidCandidate:
    """생존 가능 빈 공간(Void) 후보 1건.

    Args:
        id:               고유 식별자
        structure_type:   "lean_to" | "v_shape" | "pancake" | "pocket"
        volume_m3:        추정 부피 (m³)
        confidence:       "confirmed" | "estimated" | "low_priority"
        priority_score:   구조 우선순위 점수 (0.0 ~ 1.0)
        center_xyz:       Void 중심 3D 좌표 (m)
        entry_point:      진입 지점 3D 좌표 (m), 없으면 None
        stability_score:  구조 안정성 (0.0 위험 ~ 1.0 안전)
        access_difficulty: 접근 난이도 (0.0 쉬움 ~ 1.0 어려움)
    """
    id: str
    structure_type: str
    volume_m3: float
    confidence: str
    priority_score: float
    center_xyz: tuple[float, float, float]
    entry_point: Optional[tuple[float, float, float]]
    stability_score: float
    access_difficulty: float


@dataclass
class AnalysisResult:
    """전체 분석 파이프라인의 최종 결과.

    Args:
        void_candidates:     발견된 Void 후보 목록 (List[VoidCandidate])
        point_cloud:         DUSt3R 복원 포인트 클라우드 (N, 3) float32
        voxel_grid:          Voxel Grid (X, Y, Z) bool
        voxel_size_m:        격자 크기 (m), 기본 0.05
        images_used:         실제 분석에 사용된 이미지 수
        processing_time_sec: 전체 처리 소요 시간 (초)
    """
    void_candidates: list[VoidCandidate]
    point_cloud: np.ndarray        # (N, 3) float32
    voxel_grid: np.ndarray         # (X, Y, Z) bool
    voxel_size_m: float
    images_used: int
    processing_time_sec: float

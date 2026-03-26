"""
analysis/voxel/converter.py

Point Cloud → Voxel Grid → Void 클러스터 변환 파이프라인.

CLAUDE.md §8.1 알고리즘 순서:
  1. Point Cloud → Occupied Voxel Grid (bool 3D)
  2. 6-connectivity Flood Fill (바깥 경계 시작) → External Voxels
  3. NOT(Occupied) AND NOT(External) → Void Voxels
  4. Connected Component Labeling → Void Clusters
  5. volume_m3 = voxel_count × voxel_size³  →  >= VOID_MIN_VOLUME_M3 인 것만 반환
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from shared.config import VOXEL_SIZE_M, VOID_MIN_VOLUME_M3

logger = logging.getLogger(__name__)


# ── 내부 결과 타입 ──────────────────────────────────────────────────────────────

@dataclass
class VoxelCluster:
    """Void 클러스터 1개의 기하 정보.

    Args:
        label_id:     Connected Component 레이블 번호
        voxel_count:  해당 클러스터를 구성하는 Voxel 수
        volume_m3:    추정 부피 (m³)
        center_xyz:   클러스터 중심 3D 좌표 (m), 원래 Point Cloud 기준
    """
    label_id: int
    voxel_count: int
    volume_m3: float
    center_xyz: tuple[float, float, float]


# ── 메인 파이프라인 ─────────────────────────────────────────────────────────────

def point_cloud_to_voxel_grid(
    points: np.ndarray,
    voxel_size: float = VOXEL_SIZE_M,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Point Cloud를 Occupied Voxel Grid로 변환한다.

    Args:
        points:     (N, 3) float32 포인트 클라우드
        voxel_size: 격자 한 변의 길이 (m), 기본 VOXEL_SIZE_M

    Returns:
        grid:       (X, Y, Z) bool — True = 점유된 Voxel
        origin:     (3,) float64 — 격자 원점 (최솟값 좌표)
        voxel_size: 실제 사용된 격자 크기

    Raises:
        ValueError: points 가 비어있거나 shape 이 잘못된 경우
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points shape 이 (N, 3) 이어야 합니다. 현재: {points.shape}")
    if len(points) == 0:
        raise ValueError("포인트 클라우드가 비어있습니다.")

    origin = points.min(axis=0)
    indices = np.floor((points - origin) / voxel_size).astype(np.int32)

    # 격자 크기 결정 (+1: 경계 포함)
    grid_shape = tuple((indices.max(axis=0) + 1).tolist())
    grid = np.zeros(grid_shape, dtype=bool)
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    logger.info(
        "Voxel Grid 생성 완료: shape=%s, 점유율=%.2f%%",
        grid_shape,
        100.0 * grid.sum() / grid.size,
    )
    return grid, origin, voxel_size


def flood_fill_external(occupied: np.ndarray) -> np.ndarray:
    """6-connectivity Flood Fill 로 외부 공기와 연결된 빈 공간을 찾는다.

    격자를 1 Voxel 씩 패딩한 뒤 모든 바깥 경계 빈 칸에서 시작하여
    6방향(±X, ±Y, ±Z) BFS 를 수행한다.

    Args:
        occupied: (X, Y, Z) bool — 점유 격자

    Returns:
        external: (X, Y, Z) bool — True = 외부 공기와 연결된 빈 공간
    """
    # 패딩: 경계 바깥을 확실한 빈 공간으로 만들어 BFS 시작점 확보
    padded = np.pad(occupied, pad_width=1, mode="constant", constant_values=False)
    # 빈 공간 마스크 (점유 아님)
    free = ~padded

    # BFS — scipy label 을 이용해 연결 성분을 구하고
    # 패딩 경계(index 0 또는 max)에 닿은 성분 = external
    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1, :, 1] = True   # ±Y
    structure[:, 1, 1] = True   # ±X
    structure[1, 1, :] = True   # ±Z
    # → 6-connectivity 구조 커널

    labeled, _ = label(free, structure=structure)

    # 패딩 경계에 닿은 레이블 수집
    border_labels: set[int] = set()
    for face in (
        labeled[0, :, :], labeled[-1, :, :],
        labeled[:, 0, :], labeled[:, -1, :],
        labeled[:, :, 0], labeled[:, :, -1],
    ):
        border_labels.update(np.unique(face[face > 0]).tolist())

    # external 마스크 (패딩 제거)
    external_padded = np.isin(labeled, list(border_labels)) & free
    external = external_padded[1:-1, 1:-1, 1:-1]

    logger.info(
        "Flood Fill 완료: external=%.2f%%, void_candidate=%.2f%%",
        100.0 * external.sum() / occupied.size,
        100.0 * (~occupied[...] & ~external).sum() / occupied.size,
    )
    return external


def extract_void_clusters(
    occupied: np.ndarray,
    external: np.ndarray,
    origin: np.ndarray,
    voxel_size: float = VOXEL_SIZE_M,
    min_volume_m3: float = VOID_MIN_VOLUME_M3,
) -> list[VoxelCluster]:
    """Void Voxel 에 Connected Component Labeling 을 적용해 클러스터를 반환한다.

    Args:
        occupied:      (X, Y, Z) bool — 점유 격자
        external:      (X, Y, Z) bool — 외부 공기 격자
        origin:        (3,) 격자 원점 좌표 (m)
        voxel_size:    격자 크기 (m)
        min_volume_m3: 이 부피 미만 클러스터는 제외

    Returns:
        생존 후보 클러스터 목록 (부피 내림차순 정렬)
    """
    void_mask = ~occupied & ~external

    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1, :, 1] = True
    structure[:, 1, 1] = True
    structure[1, 1, :] = True

    labeled, num_features = label(void_mask, structure=structure)
    logger.info("Connected Component Labeling: %d 개 클러스터 발견", num_features)

    voxel_volume = voxel_size ** 3
    clusters: list[VoxelCluster] = []

    for lbl in range(1, num_features + 1):
        mask = labeled == lbl
        count = int(mask.sum())
        volume = count * voxel_volume

        if volume < min_volume_m3:
            continue

        # 클러스터 중심: 인덱스 평균 → 실제 좌표
        idx = np.argwhere(mask)
        center_idx = idx.mean(axis=0)
        center_xyz = tuple((origin + (center_idx + 0.5) * voxel_size).tolist())

        clusters.append(VoxelCluster(
            label_id=lbl,
            voxel_count=count,
            volume_m3=round(volume, 4),
            center_xyz=center_xyz,
        ))

    clusters.sort(key=lambda c: c.volume_m3, reverse=True)
    logger.info(
        "생존 후보 클러스터: %d 개 (>= %.1f m³)",
        len(clusters), min_volume_m3,
    )
    return clusters


def run_voxel_pipeline(
    points: np.ndarray,
    voxel_size: float = VOXEL_SIZE_M,
    min_volume_m3: float = VOID_MIN_VOLUME_M3,
) -> tuple[list[VoxelCluster], np.ndarray, np.ndarray]:
    """Point Cloud 를 받아 Void 클러스터 목록을 반환하는 통합 파이프라인.

    Args:
        points:        (N, 3) float32 포인트 클라우드
        voxel_size:    격자 크기 (m)
        min_volume_m3: 생존 판정 최소 부피 (m³)

    Returns:
        clusters:  생존 후보 VoxelCluster 목록 (부피 내림차순)
        occupied:  (X, Y, Z) bool 점유 격자
        external:  (X, Y, Z) bool 외부 공기 격자
    """
    occupied, origin, voxel_size = point_cloud_to_voxel_grid(points, voxel_size)
    external = flood_fill_external(occupied)
    clusters = extract_void_clusters(
        occupied, external, origin, voxel_size, min_volume_m3
    )
    return clusters, occupied, external

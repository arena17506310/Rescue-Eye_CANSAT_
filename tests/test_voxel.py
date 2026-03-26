"""
tests/test_voxel.py

Phase 1 완료 기준:
  1×1×2m 박스 속 빈 공간 입력 시 volume ≈ 2.0m³ (오차 5% 이내)

추가 테스트:
  - 박스 껍데기만 있을 때 내부 Void 탐지
  - 작은 공동(< 1.5m³)은 필터링되는지
  - 포인트 클라우드 입력 검증 (빈 배열 / 잘못된 shape)
"""

import sys
import os
import logging
import numpy as np
import pytest

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.voxel.converter import (
    point_cloud_to_voxel_grid,
    flood_fill_external,
    extract_void_clusters,
    run_voxel_pipeline,
    VoxelCluster,
)
from shared.config import VOXEL_SIZE_M, VOID_MIN_VOLUME_M3

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


# ── 헬퍼 ────────────────────────────────────────────────────────────────────────

def make_hollow_box(
    x_m: float, y_m: float, z_m: float,
    voxel_size: float = VOXEL_SIZE_M,
    thickness: int = 1,
) -> np.ndarray:
    """지정 크기의 속이 빈 직육면체 껍데기 포인트 클라우드를 생성한다.

    내부가 완전히 고립된 Void 가 되도록 6면 벽을 모두 채운다.

    Args:
        x_m, y_m, z_m: 박스 외부 치수 (m)
        voxel_size:    격자 크기 (m)
        thickness:     벽 두께 (voxel 단위)

    Returns:
        (N, 3) float32 포인트 클라우드
    """
    nx = int(round(x_m / voxel_size))
    ny = int(round(y_m / voxel_size))
    nz = int(round(z_m / voxel_size))

    occupied = np.zeros((nx, ny, nz), dtype=bool)

    # 6면 벽 채우기
    occupied[:thickness, :, :]  = True   # -X 면
    occupied[-thickness:, :, :] = True   # +X 면
    occupied[:, :thickness, :]  = True   # -Y 면
    occupied[:, -thickness:, :] = True   # +Y 면
    occupied[:, :, :thickness]  = True   # -Z 면
    occupied[:, :, -thickness:] = True   # +Z 면

    idx = np.argwhere(occupied).astype(np.float32)
    points = idx * voxel_size
    return points


# ── 테스트 ───────────────────────────────────────────────────────────────────────

class TestPointCloudToVoxelGrid:
    def test_basic_conversion(self):
        """포인트가 있는 곳에 Voxel 이 생성되는지 확인한다."""
        points = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]], dtype=np.float32)
        grid, origin, vs = point_cloud_to_voxel_grid(points, voxel_size=0.05)
        assert grid.any(), "점유된 Voxel 이 없습니다."
        assert grid.dtype == bool

    def test_empty_points_raises(self):
        """빈 포인트 배열은 ValueError 를 발생시켜야 한다."""
        with pytest.raises(ValueError, match="비어있습니다"):
            point_cloud_to_voxel_grid(np.empty((0, 3), dtype=np.float32))

    def test_wrong_shape_raises(self):
        """(N, 2) 배열은 ValueError 를 발생시켜야 한다."""
        with pytest.raises(ValueError, match="shape"):
            point_cloud_to_voxel_grid(np.ones((10, 2), dtype=np.float32))


class TestFloodFill:
    def test_external_not_inside_solid_box(self):
        """꽉 찬 박스에서는 내부 Void 가 없어야 한다."""
        occupied = np.ones((5, 5, 5), dtype=bool)
        external = flood_fill_external(occupied)
        # occupied 가 True 인 곳 external 은 False
        assert not external[occupied].any()

    def test_hollow_box_has_internal_void(self):
        """속이 빈 박스에서는 내부가 Void 로 분류되어야 한다."""
        occupied = np.ones((7, 7, 7), dtype=bool)
        occupied[2:5, 2:5, 2:5] = False   # 내부 3×3×3 제거
        external = flood_fill_external(occupied)
        void = ~occupied & ~external
        assert void[2:5, 2:5, 2:5].all(), "내부 공간이 Void 로 탐지되지 않았습니다."

    def test_open_box_no_internal_void(self):
        """한 면이 열린 박스는 내부가 외부와 연결되어 Void 가 없어야 한다."""
        occupied = np.ones((7, 7, 7), dtype=bool)
        occupied[2:5, 2:5, 2:5] = False
        # 내부(z=4)에서 바깥까지 이어지는 채널 개방: 벽 2개 모두 제거
        occupied[3, 3, 5] = False   # 첫 번째 벽 voxel
        occupied[3, 3, 6] = False   # 두 번째 벽 voxel (+Z 면 경계)
        external = flood_fill_external(occupied)
        void = ~occupied & ~external
        # 내부가 외부와 연결되었으므로 Void 가 없어야 함
        assert not void[2:5, 2:5, 2:5].any(), \
            "열린 박스 내부가 잘못 Void 로 분류되었습니다."


class TestExtractVoidClusters:
    def test_small_void_filtered_out(self):
        """최소 부피 미만 클러스터는 반환되지 않아야 한다."""
        points = make_hollow_box(0.3, 0.3, 0.3)   # 내부 부피 << 1.5m³
        clusters, occupied, external = run_voxel_pipeline(
            points, voxel_size=VOXEL_SIZE_M, min_volume_m3=VOID_MIN_VOLUME_M3
        )
        assert len(clusters) == 0, \
            f"작은 공동이 필터링되지 않았습니다: {clusters}"

    def test_cluster_has_positive_volume(self):
        """탐지된 클러스터 부피는 모두 양수여야 한다."""
        points = make_hollow_box(2.0, 2.0, 2.0)
        clusters, _, _ = run_voxel_pipeline(points)
        for c in clusters:
            assert c.volume_m3 > 0


class TestPhase1CompletionCriteria:
    """Phase 1 완료 기준: 1×1×2m 박스 → volume ≈ 2.0m³ (오차 5% 이내)"""

    TARGET_VOLUME_M3 = 1.0 * 1.0 * 2.0   # 2.0m³
    TOLERANCE = 0.05                       # 5%

    def test_1x1x2_box_volume(self):
        """1×1×2m 속이 빈 박스 내부 부피가 2.0m³ ±5% 이내인지 검증한다."""
        # 내부 1×1×2m 확보: 벽 두께(1 voxel = 0.05m) × 양쪽 = 0.1m 추가
        # → 외부 치수 1.1×1.1×2.1m → 내부 격자 20×20×40 → 2.0m³
        points = make_hollow_box(1.1, 1.1, 2.1, voxel_size=VOXEL_SIZE_M)
        clusters, _, _ = run_voxel_pipeline(
            points, voxel_size=VOXEL_SIZE_M, min_volume_m3=1.0   # 임계값 낮춰서 탐지
        )

        assert len(clusters) >= 1, \
            "1×1×2m 박스 내부 Void 가 탐지되지 않았습니다."

        largest = clusters[0]
        error_rate = abs(largest.volume_m3 - self.TARGET_VOLUME_M3) / self.TARGET_VOLUME_M3

        print(f"\n[Phase 1 검증]")
        print(f"  탐지된 최대 클러스터 부피: {largest.volume_m3:.4f} m³")
        print(f"  목표 부피:                {self.TARGET_VOLUME_M3:.4f} m³")
        print(f"  오차율:                   {error_rate * 100:.2f}%")
        print(f"  허용 오차:                {self.TOLERANCE * 100:.1f}%")
        print(f"  판정: {'PASS ✓' if error_rate <= self.TOLERANCE else 'FAIL ✗'}")

        assert error_rate <= self.TOLERANCE, (
            f"부피 오차 {error_rate*100:.2f}% 가 허용치 {self.TOLERANCE*100:.1f}% 초과. "
            f"측정값={largest.volume_m3:.4f}m³, 목표={self.TARGET_VOLUME_M3:.4f}m³"
        )

    def test_result_is_sorted_by_volume_descending(self):
        """클러스터 목록이 부피 내림차순으로 정렬되어 있어야 한다."""
        points = make_hollow_box(2.0, 2.0, 2.0)
        clusters, _, _ = run_voxel_pipeline(points, min_volume_m3=0.01)
        volumes = [c.volume_m3 for c in clusters]
        assert volumes == sorted(volumes, reverse=True), \
            "클러스터가 부피 내림차순으로 정렬되지 않았습니다."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

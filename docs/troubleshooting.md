# RESCUE-EYE 자주 발생하는 문제 및 해결법
# 에러 디버깅 시 참조

---

| 문제 | 원인 | 해결법 |
|------|------|--------|
| Bluetooth 연결 안 됨 | baud rate 9600 (공장 기본값) | 지상 점검 모듈로 115200 설정 |
| GPS 파싱 실패 | $GPGGA / $GNGGA 혼용 | 두 형식 모두 처리하는 파서 작성 |
| 이미지 전부 Blur 처리됨 | 임계값 오설정 | `BLUR_THRESHOLD` 낮추기 (100→50) |
| DA3 포즈 추정 불안정 | 이미지 overlap 부족 또는 고도 문제 | COLMAP fallback 자동 발동 확인, 촬영 간격 줄이기 |
| DA3 confidence 전부 낮음 | 이미지 품질 문제 | `DA3_CONFIDENCE_THRESHOLD` 낮추기 (0.5→0.3) |
| DA3 glb export 실패 | gsplat 미설치 | `pip install --no-build-isolation git+...gsplat.git` |
| DA3 OOM | DA3_BATCH_SIZE 과다 | `DA3_BATCH_SIZE = 4` 유지, Giant → Large로 변경 |
| COLMAP fallback도 실패 | 이미지 overlap 부족 | 촬영 간격 줄이기 또는 해상도 높이기 |
| Voxel 부피 오차 큼 | 스케일 미보정 | DA3 metric depth로 Point Cloud 스케일 재보정 |
| 짐벌 진동 | PID 게인 과다 | `gimbal.py` KP값 0.5배 감소 |
| RPi ↔ ATmega128 통신 불안정 | 5V↔3.3V 직결 | 레벨 컨버터 회로 점검 |
| Florence-2가 입구 못 찾음 | 일반 프롬프트 사용 | `domain/prompts.py` 도메인 프롬프트로 교체 |
| SAM2 분할 결과 불안정 | 단일 프레임 모드 사용 | 연속 프레임 video mode로 전환 |

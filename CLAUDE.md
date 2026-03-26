# RESCUE-EYE — CLAUDE.md v6.0
# Claude Code 전용 운용 지침 (토큰 최적화 버전)

> **이 파일은 Claude Code가 모든 작업 전에 반드시 읽어야 하는 유일한 진실의 원천이다.**
> 이 파일에 없는 내용을 임의로 추정하지 말 것. 불확실하면 작업을 멈추고 확인을 요청하라.
> **상세 문서는 `docs/` 폴더에 분리되어 있다. 필요할 때만 읽어라.**

---

## 1. MISSION BRIEF

RESCUE-EYE는 건물 붕괴 현장에 투하되는 캔위성(CanSat)이 낙하 중 촬영한 사진으로
**생존 가능한 빈 공간(Void)**을 탐지하고 3D 구조 지도를 생성하는 재난 대응 시스템이다.

- 대회: 2026 KAIST 캔위성 체험·경연대회 (슬기부)
- Void 판정 기준: 성인 1명 생존 최소 공간 **1.5m³** 이상

---

## 2. 시스템 3줄 요약

1. **낙하 중**: RPi Zero 2W가 촬영 + SD저장, ATmega128이 Bluetooth로 텔레메트리 전송
2. **착륙 후**: SD카드 물리 회수 → RTX 3080 데스크톱에 직접 삽입 (네트워크 없음)
3. **분석**: COLMAP → DA3 → SAM2 → Florence-2 → Voxel → 3D 구조 우선순위 지도

---

## 3. 디렉토리 구조

```
rescue-eye/
├── CLAUDE.md                 ← 이 파일 (삭제/이동 금지)
├── .claudeignore             ← Claude Code 탐색 제외 목록
├── docs/                     ← 상세 문서 (필요할 때만 읽기)
│   ├── hardware.md           # 하드웨어 아키텍처 + 연결 주의사항
│   ├── domain.md             # 붕괴 현장 도메인 지식 + 프롬프트
│   ├── algorithms.md         # Voxel 파이프라인 + 융합 판정 로직 + 데이터 모델
│   ├── setup.md              # Phase 0 환경 세팅 명령어
│   └── troubleshooting.md    # 자주 발생하는 문제 및 해결법
│
├── cansat/                   ← RPi Zero 2W 탑재 코드 (Python)
│   ├── main.py               # 메인 루프: 촬영 + 짐벌 + Blur + SD저장
│   ├── camera.py             # Global Shutter 카메라 제어 (picamera2)
│   ├── gimbal.py             # MPU6050 읽기 + Servo PID 제어
│   └── blur_check.py         # Laplacian 분산 계산 → 수치 반환
│
├── cansat_firmware/          ← ATmega128 펌웨어 (C/C++)
│   ├── main.c
│   ├── bluetooth.c / .h      # 115200bps 송신
│   ├── imu.c / .h            # EBIMU-9DOFV5 파싱
│   ├── gps.c / .h            # $GPGGA / $GNGGA 파싱
│   └── telemetry.c / .h      # 패킷: [START][TYPE][DATA][CHECKSUM][END]
│
├── ground/                   ← 노트북 지상국 코드 (Python)
│   ├── main.py
│   ├── receiver.py            # Bluetooth 시리얼 수신 (pyserial)
│   ├── parser.py              # 텔레메트리 패킷 파싱
│   ├── blur_logger.py         # Blur 수치 JSON 저장
│   └── monitor_ui.py          # 실시간 모니터링 화면
│
├── analysis/                 ← RTX 3080 로컬 분석 코드 (Python)
│   ├── main.py                # python main.py --sd /path --blur_log /path
│   ├── preprocess/
│   │   ├── selector.py        # Blur 기반 이미지 선별
│   │   └── loader.py
│   ├── pipeline/
│   │   ├── reconstruct.py     # COLMAP SfM
│   │   ├── depth.py           # DA3 Dense Depth
│   │   ├── segment.py         # SAM2 인스턴스 분리
│   │   ├── detect.py          # Florence-2 입구 탐지
│   │   └── fuse.py            # 4모델 융합 판정
│   ├── voxel/
│   │   ├── converter.py       # Point Cloud → Voxel Grid
│   │   ├── analyzer.py        # Void 클러스터링 + 부피(m³)
│   │   └── scorer.py          # 우선순위 점수
│   └── visualize/
│       └── renderer.py        # 3D 지도 렌더링
│
├── domain/
│   ├── prompts.py             # Florence-2 도메인 프롬프트 (→ docs/domain.md 참조)
│   ├── vocabulary.py          # 붕괴 구조 용어 사전
│   └── training_data/
│
├── shared/
│   ├── config.py              # 전역 설정값 (수치 하드코딩 금지)
│   └── data_models.py         # 공통 데이터 구조 (dataclass)
│
└── tests/
    ├── test_blur.py
    ├── test_voxel.py
    ├── test_pipeline.py
    └── mock_data/
```

---

## 4. 언어 및 개발 환경

| 노드 | 언어 | 핵심 라이브러리 |
|------|------|---------------|
| ATmega128 (기본 키트) | C/C++ (AVR-GCC) | — (8비트 MCU, Python 불가) |
| RPi Zero 2W (탑재체) | Python 3.10+ | picamera2, smbus2, RPi.GPIO |
| 노트북 (지상국) | Python 3.10+ | pyserial, numpy, opencv-python |
| RTX 3080 데스크톱 | Python 3.10+ | PyTorch(CUDA), open3d, AI 모델 전체 |

---

## 5. 핵심 설정값 (config.py 요약)

```
촬영 간격: 0.5s | 짐벌 목표: -65° | Blur 임계값: 100.0
BT: 115200bps (공장기본 9600→변경필수) | GPS: 9600bps | IMU: 115200bps
Voxel: 0.05m | Void 최소 부피: 1.5m³ | Depth 최소: 0.5m
DA3: Large (~4GB) | SAM2: Large | DA3 배치: 4
```

---

## 6. VRAM 관리 (RTX 3080 = 10GB)

순차 실행: COLMAP(~1GB) → DA3-Large(~4GB) → SAM2(~2GB) → Florence-2(~2GB) = ~9GB
**모델은 프로그램 시작 시 한 번만 로드. 요청마다 재로드 절대 금지.**

---

## 7. 개발 Phase

**현재 상태: Phase 0 (환경 세팅 미완료, 코드 없음)**
이전 Phase 미완료 시 다음 Phase 시작 금지.

| Phase | 내용 | 완료 기준 |
|-------|------|----------|
| 0 | 환경 세팅 (→ `docs/setup.md`) | `torch.cuda.is_available() == True` |
| 1 | Voxel 변환 엔진 | 1×1×2m 박스 → volume ≈ 2.0m³ (오차 5%) |
| 2 | AI 파이프라인 | preprocess→COLMAP→DA3→SAM2→Florence-2→fuse→voxel→visualize |
| 3 | 캔위성 탑재 소프트웨어 | RPi + ATmega128 펌웨어 |
| 4 | 지상국 모니터링 | Bluetooth 수신 + 실시간 UI |
| 5 | 결과 시각화 | 3D 복셀 지도 (빨강/주황/노랑) |

---

## 8. 코드 작성 규칙

**Python**: 3.10+, 타입 힌트 필수, 모든 함수 docstring, `logging`만 사용(`print()` 금지), 설정값은 `shared/config.py`에서만 관리

**C/C++ (ATmega128)**: AVR-GCC, 인터럽트 최소화, UART 버퍼 오버플로 방지, 패킷: `[START][TYPE][DATA][CHECKSUM][END]`

**AI 추론**: 반드시 try/except로 OOM 처리, `torch.cuda.empty_cache()` 호출

---

## 9. 팀원 역할

| 팀원 | 담당 |
|------|------|
| 장시우 (팀장) | `analysis/pipeline/`, `analysis/voxel/`, `domain/` |
| 황준화 | `ground/`, `cansat_firmware/`, `shared/` |
| 김준성 | `cansat/`, `cansat_firmware/` 하드웨어 연동 |

---

## 10. 상세 문서 참조 가이드

| 작업 중인 영역 | 읽어야 할 문서 |
|---------------|--------------|
| 하드웨어/통신/센서 | `docs/hardware.md` |
| AI 모델/프롬프트/도메인 지식 | `docs/domain.md` |
| Voxel/융합 판정/데이터 모델 | `docs/algorithms.md` |
| 환경 세팅/설치 | `docs/setup.md` |
| 에러/디버깅 | `docs/troubleshooting.md` |

> **이 문서들을 미리 읽지 말 것. 해당 영역 작업 시에만 읽어라.**

---

## Compaction 시 보존할 것

- 수정한 파일 목록과 변경 내용
- 현재 Phase와 진행 상태
- 발생한 에러와 해결 여부
- 작업 중인 파일 경로

---

*RESCUE-EYE CLAUDE.md v6.0 — 2026 KAIST 캔위성 경연대회*
*이 파일은 프로젝트 루트 rescue-eye/CLAUDE.md 에 위치해야 한다.*

# RESCUE-EYE 붕괴 현장 도메인 지식
# 이 문서는 domain/, analysis/pipeline/ 작업 시 참조

---

## 핵심: RESCUE-EYE의 차별점은 붕괴 구조 전문 용어로 최적화된 AI다.

Claude Code는 아래 도메인 지식을 기반으로 프롬프트, 데이터, 판정 로직을 작성해야 한다.

---

## 1. 생존 공간 구조 유형

```
Lean-to (기대기형)
  슬래브가 무너지며 다른 잔해에 비스듬히 기대어
  내부에 삼각형 단면의 생존 공간 형성.
  구조대 용어: "lean-to collapse", "inclined slab void"

V-shape (V자형)
  천장/슬래브가 V자로 꺾여
  양쪽 경사면 아래 공간 형성.
  구조대 용어: "V-shape collapse", "inverted V void"

Pancake (팬케이크형)
  여러 층 슬래브가 수평으로 겹침.
  층 사이 얇은 틈새에 생존 공간 가능.
  구조대 용어: "pancake collapse", "floor sandwich void"

Void pocket (고립 공간)
  잔해 더미 내부에 완전히 고립된 공간.
  외부에서 입구 안 보임.
  → Voxel 내부 탐색(flood fill)으로만 발견 가능.
  구조대 용어: "trapped void", "buried air pocket"
```

---

## 2. Florence-2 도메인 프롬프트 (`domain/prompts.py`)

```python
# 생존 공간 입구 탐지 — Zero-shot 프롬프트 목록
VOID_ENTRY_PROMPTS = [
    # 구조 형태 기반
    "lean-to void",
    "V-shape gap between concrete slabs",
    "triangular opening in rubble",
    "pancake void space",
    "survivable space under debris",
    "inclined slab cavity",

    # 시각적 특징 기반
    "dark opening in collapsed structure",
    "hole between concrete slabs",
    "gap under fallen ceiling",
    "tunnel through rubble",
    "accessible void entrance",
    "shadow indicating deep cavity",

    # 구조 부재 인식
    "tilted concrete slab",
    "leaning wall fragment",
    "collapsed floor panel",
    "broken column support",
    "debris pile with hollow interior",
]

# SAM 구조물 분류 레이블
STRUCTURAL_LABELS = [
    "concrete slab",       # 콘크리트 슬래브
    "floor panel",         # 바닥판
    "wall fragment",       # 벽체 파편
    "column stub",         # 기둥 잔재
    "rebar mesh",          # 철근
    "debris pile",         # 잔해 더미
    "ground surface",      # 지면
]

# 불안정 구조 경고 키워드
STABILITY_RISK_KEYWORDS = [
    "unstable overhang",
    "precariously balanced slab",
    "cracked support column",
    "secondary collapse risk",
]
```

---

## 3. 구조 안정성 판단 기준

```python
def classify_stability(tilt_deg: float, thickness_m: float, support_count: int) -> float:
    """
    SAM2로 분리된 각 구조물의 안정성 점수 산출.
    Returns: 0.0 (매우 위험) ~ 1.0 (안전)
    """
    tilt_score      = max(0.0, 1.0 - tilt_deg / 45.0)       # 45° 이상 = 위험
    thickness_score = min(1.0, thickness_m / 0.3)            # 30cm 이상 = 안전
    support_score   = min(1.0, support_count / 2.0)          # 지지점 2개 이상 = 안전
    return tilt_score * 0.4 + thickness_score * 0.3 + support_score * 0.3
```

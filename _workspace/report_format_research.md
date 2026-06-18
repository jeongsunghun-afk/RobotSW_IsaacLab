# 작업 세션 보고서 양식 리서치 — 베스트 프랙티스 & 추천 설계

> **결론 먼저 (BLUF):** 현재 양식의 문제는 "섹션이 너무 많고 평면적"인 것. 해법은 **섹션 수를 줄이고 + 위계로 접는 것(progressive disclosure)**이다. 한눈에 보는 요약(at-a-glance)을 맨 위에 두고, **변경→효과 인과를 표로** 드러내고, 상세 근거는 `<details>`로 접어 아래에 둔다.

## (a) 영역별 핵심 패턴

### 1. Engineering Decision Records / ADR
- **Context → Decision → Consequences 3단 구조**
- **Consequences는 정직성 강제 장치** — 긍정·부정 결과를 같은 섹션에 (RL 트레이드오프 기록에 유용)
- **결정은 수정하지 않고 supersede** — deprecated/superseded by 상태만 갱신
- **Status 마커(Proposed/Accepted/Deprecated)** → ✅/⚠️/❌로 변환

### 2. Changelog (Keep a Changelog) & Conventional Commits
- **변경을 유형별로 그룹핑** — Added/Changed/Fixed/Removed → 추가(reward/obs/env)/변경(hyperparam/network)/수정(bug/롤백)
- **"사람을 위해" + 항목당 1~2문장** — 길면 링크/접기로 위임
- **"메트릭/학습 결과에 영향을 주면 기록"** — 무영향 리팩토링 생략 가능
- **Conventional Commits 접두사**(feat:/fix:/refactor:)로 변경 의도 타입화

### 3. Pull Request / 변경 요약
- **What / Why / How 3분할 — How(=diff)는 최소화**
- **검증 방법(테스트/스모크/메트릭) 필수 칸**
- **before/after 시각 증거** — RL은 메트릭 before→after 표/그래프
- **템플릿이 행동을 유도** — 빈 칸이 효과 기록을 강제

### 4. Executive Summary / BLUF
- **BLUF = 결론을 맨 앞에** (근거가 결론을 뒤따름)
- **Inverted Pyramid** — 중요 정보 꼭대기, 아래로 세부
- **요약은 thesis처럼 단순·간결** — "한 줄 요약 + 3불릿"

### 5. Experiment / RL 실험 로그 ★우리 세션의 척추
- **hypothesis → change → result → next 사이클** — 세션 1건 = 사이클 1~N개
- **하이퍼파라미터·설정 전량 기록** (재현성)
- **사람이 읽는 실험 라벨/태그** (baseline/reward-v3 등)
- **artifact provenance** — git SHA + 로그 경로 + WandB run 링크
- **메트릭 판정(Green/Yellow/Red)** — log-analyzer 판정과 일치

### 6. 가독성 / 스캐너빌리티
- **사용자는 읽지 않고 스캔한다(Nielsen)** — heading은 표지판
- **Bold는 절제 — 핵심어에만**
- **표 = 비교·인과의 최적 도구** (before/after, 변경/효과)
- **제한된 포맷 팔레트를 일관되게**
- **정보 밀도 ↔ 한눈에 보기 = 레이어링으로 해소** (`<details>`)

## (b) 추천 패턴 Top 10

| # | 패턴 | 왜 |
|---|------|-----|
| 1 | **BLUF 한 줄 결론 + 3불릿** 최상단 | 0.5초 안에 "뭐 했나" 파악 |
| 2 | **`변경 → 이유 → 효과(before→after)` 표** | 인과를 눈으로 — 핵심 가치 |
| 3 | **`<details>` 접기로 상세 근거 레이어링** | "짧지만 근거 접근 가능" |
| 4 | **상태 마커 표준화 (✅⚠️❌🔄)** — 상태에만 | 스캔으로 즉시 인식 |
| 5 | **hypothesis → change → result → next 블록** | RL 세션의 본질 = 실험 |
| 6 | **변경을 유형 태그로 분류** (추가/변경/수정/롤백) | 스캔 가능한 한 줄들 |
| 7 | **What·Why 집중, How(=diff) 최소화** | 길이 폭증 방지 |
| 8 | **설정·하이퍼파라미터 전량 + git SHA/run 링크** (접기 안에) | 재현성 |
| 9 | **메트릭 Green/Yellow/Red 판정** | 수치 나열 대신 상태 요약 |
| 10 | **최상위 섹션 ≤5개로 강제** | "섹션 과다·평면" 직접 해결 |

> 1·2·3·10이 가독성 문제 직접 해결 핵심. 나머지는 RL 도메인 적합성.

## (c) 스켈레톤 (상세는 원본 리서치 참조)

- **A 표준형**: BLUF → 1.변경→효과 표 → 2.결과/메트릭(+details 접기) → 3.다음 액션
- **B 실험 사이클형**: BLUF → 실험 로그(가설/변경/결과/다음 반복) → details
- **C 미니멀형**: 제목+BLUF + 변경표 + 다음

## 상태 마커 가이드 (남용 금지, 상태에만)
| 마커 | 의미 |
|---|---|
| ✅ | 완료/성공/검증됨 |
| ⚠️ | 부분 성공/주의/미세 효과 |
| ❌ | 실패/롤백/폐기 |
| 🔄 | 진행중/다음 이터레이션 대기 |
| 🟢🟡🔴 | 메트릭 종합 판정 (log-analyzer 체계) |

## ⚠️ 미해결 리스크 (구현 전 검증 필요)
- **Notion MCP 렌더링**: `<details>` 접기, `> [!NOTE]` callout이 `notion-create-pages`에서 렌더링 안 될 수 있음. 현재 템플릿이 `##/###`만 허용하는 이유. 이모지/표/blockquote는 안전. → 설계 결정에서 검증 필요.

## Sources
ADR(Nygard, adr.github.io, Fowler), Keep a Changelog 1.1.0, Conventional Commits, DeployHQ/freeCodeCamp/Pragmatic Engineer PR, BLUF/Inverted Pyramid(Wikipedia), W&B/Neptune 실험추적, Nielsen/IxDF 가독성

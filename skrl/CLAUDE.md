# skrl 라이브러리 컨텍스트

## 개요

[skrl](https://skrl.readthedocs.io/) 의 **커스텀 포크**. IsaacLab 환경과 AMP 알고리즘 통합을 위해
원본 라이브러리를 수정한 버전이다. `pip install -e skrl/` 로 개발 모드 설치.

## 디렉토리 구조

```
skrl/
├── skrl/                          # 메인 라이브러리
│   ├── agents/torch/
│   │   ├── amp/                   # AMP 알고리즘 ← 주요 수정 대상
│   │   │   ├── amp.py             # AMP 에이전트 클래스 + compute_gae()
│   │   │   └── amp_cfg.py         # AMP_CFG 기본값
│   │   └── ppo/                   # PPO (참조용)
│   ├── envs/                      # IsaacLab 환경 래퍼
│   ├── memories/torch/            # RandomMemory (AMP 경험 버퍼)
│   ├── models/torch/              # Gaussian/Deterministic 정책 모델
│   ├── trainers/torch/            # Sequential/Parallel 트레이너
│   └── utils/                     # Runner, Spaces 유틸
├── pyproject.toml
└── CLAUDE.md
```

## 주요 파일

| 파일 | 역할 |
|------|------|
| `skrl/agents/torch/amp/amp.py` | AMP 에이전트 메인 로직 (pre_interaction, act, record_transition, update) |
| `skrl/agents/torch/amp/amp_cfg.py` | AMP 기본 하이퍼파라미터 딕셔너리 |
| `skrl/envs/torch/wrappers/isaaclab.py` | IsaacLab → skrl 환경 래퍼 (AMP obs 처리 포함) |
| `skrl/utils/torch/runner.py` | YAML cfg 기반 에이전트/트레이너 자동 구성 |

## 수정 가이드라인

- **수정 가능**: `skrl/agents/torch/amp/` — AMP 로직 개선, 버그 수정
- **수정 시 주의**: `skrl/envs/` — IsaacLab 래퍼는 AMP obs 파이프라인과 결합되어 있음
- **수정 지양**: `skrl/memories/`, `skrl/trainers/` — 코어 인프라, 변경 시 전체 영향

## AMP 에이전트 구조 요약

```python
class AMP(Agent):
    # act(): 정책 순전파 + AMP 관측 버퍼 업데이트
    # record_transition(): 경험 저장 (task + AMP 버퍼 분리)
    # pre_interaction(): AMP 참조 모션 샘플링
    # update(): PPO loss + Discriminator loss 동시 업데이트
```

Discriminator 입력: `amp_obs_t` + `amp_obs_t+1` (연결, 86-dim for 43×2)

## 알고리즘 설정 (환경별 YAML)

환경별 하이퍼파라미터는 환경 디렉토리의 `agents/skrl_amp_cfg.yaml`에서 관리.
skrl 코어의 `AMP_CFG`는 기본값이며, YAML이 이를 오버라이드한다.

## Worker 매핑

| 작업 | 담당 Worker |
|------|------------|
| AMP 에이전트 로직 수정 | `loss-worker` (update 메서드) |
| Actor/Critic/Discriminator 아키텍처 | `network-worker` |
| 하이퍼파라미터 튜닝 | `hyperparam-worker` |
| 디버그 | `il-debug-worker` |

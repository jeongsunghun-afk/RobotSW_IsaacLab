# IsaacLab Project Guide

## 프로젝트 미션
Go2(+Neck Module) 및 R_Skeleton의 **자연스럽고 강건한 보행** 구현.
- 강건성: 속도 명령에 정확히 추종, 외란에 강인
- 자연스러움: AMP및 모방학습 기반 dog motion 모방 (발 충격 최소화, 부드러운 동작)
- 확장 순서: Go2 → Go2+Neck → R_Skeleton

## 실행 환경
- Isaac Sim 기반 GPU 가속 RL 프레임워크
- Python 3.10+ (Isaac Sim 4.5: 3.10 / 5.0+: 3.11)
- PyTorch 2.7+ (CUDA 12.8)

## 주요 명령어

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task R_Skeleton-AMP-v0 --num_envs 4096

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task R_Skeleton-AMP-v0 --num_envs 32

./isaaclab.sh -f    # 코드 포맷 (ruff)
./isaaclab.sh -t    # 테스트
./isaaclab.sh -i    # 패키지 설치 (개발 모드)
```

## 프로젝트 구조

```
source/isaaclab_tasks/isaaclab_tasks/direct/
├── go2/            → go2_wtw_context.md
├── go2_amp/        → go2_amp_context.md
├── R_Skeleton/     → r_skeleton_context.md
└── R_Skeleton_amp/ → r_skeleton_amp_context.md
rsl_rl/             → rsl_rl_algorithms.md
```

## Auto-routing 규칙

사용자 요청 수신 시 아래 규칙으로 자동 분류한다.

### 라우팅 테이블
| 분류 키워드 | 호출 대상 |
|-----------|---------|
| ENV (reward, obs, 보상, 환경, cfg, 관절, 발) | `/env-coordinator` |
| ALGO (network, loss, PPO, AMP, discriminator, 하이퍼파라미터) | `/algo-coordinator` |
| 리서치, 논문, 최신 방법, survey | `/research-team` |
| 학습 루프, 이터레이션, 전체 파이프라인 | `/locomotion-loop` |
| 보고, report, notion | `/report` |

### Coordinator 역할
- **env-coordinator**: ENV worker(obs/reward/cfg)들을 병렬 dispatch 후 validate-code 자동 실행
- **algo-coordinator**: ALGO worker(network/loss/hyperparam)들을 병렬 dispatch 후 validate-code 자동 실행

## Worker 레지스트리

### 환경 계열 (Haiku)
| Worker | context 파일 | 불변 규칙 |
|--------|-------------|-----------|
| obs-worker | `*_env.py`의 `_get_observations()` (~50줄) | obs_space 숫자 = cat 결과 크기 |
| reward-worker | `*_env.py`의 `_get_rewards()` + cfg scales (~60줄) | 페널티=음수, 새 버퍼→_reset_idx 초기화 |
| cfg-worker | `*_env_cfg.py` 전체 (~100줄) | @configclass, 타입 일치 |

### 알고리즘 계열 (Haiku)
| Worker | context 파일 | 불변 규칙 |
|--------|-------------|-----------|
| network-worker | `actor_critic.py` 또는 `amp_discriminator.py`의 `__init__`+`forward` (~60줄) | input/output shape 일치 |
| loss-worker | `ppo_amp.py`의 `update()` 메서드 (~80줄) | .detach() 위치 주의 |
| hyperparam-worker | `rsl_rl_ppo_cfg.py` 전체 (~50줄) | 타입 일치, 범위 합리성 |

### 검증 계열
| Worker | 모델 | context |
|--------|------|---------|
| validate-code | haiku | 체크리스트 A~D + git diff |
| validate-method | sonnet | 체크리스트 E~H + WebSearch |

### 분석/보고 계열
| Worker | 모델 | context |
|--------|------|---------|
| log-analyzer | haiku | 분석 지침 + 로그 N줄 |
| motion-analyzer | haiku | `motion_loader.py` 일부 |
| report-worker | sonnet | 보고서 템플릿 + Notion target |

### 디버그 계열 (Sonnet)
| Worker | 역할 | 읽는 파일 |
|--------|------|-----------|
| wtw-debug-worker | WTW 코드 레벨 분석 | `go2_env_cfg.py`, `go2_wtw_env.py` |
| amp-debug-worker | AMP 코드 레벨 분석 | `rsl_rl_ppo_cfg.py`, `ppo_amp.py`, `amp_discriminator.py` |

### 리서치 팀 계열 (Opus)
| 에이전트 | 역할 | 도구 |
|---------|------|------|
| research-surveyor | Gemini CLI + WebSearch 조사 | Bash(gemini), WebSearch |
| research-advisor | 적용 가능성 판별 (교수 역할) | Read, WebSearch |

### 학습 루프 계열 (Sonnet)
| 에이전트 | 역할 | 입력/출력 |
|---------|------|---------|
| training-evaluator | 로그 + 로봇 상태 평가 | 로그 경로 → eval_iter_{N}.md |
| feedback-generator | 평가 → 개선 피드백 변환 | eval_iter_{N}.md → feedback_iter_{N}.md |

## 역할 커맨드 (직접 호출용)
- `/env-coordinator`  : ENV 작업 오케스트레이션
- `/algo-coordinator` : ALGO 작업 오케스트레이션
- `/experiment`       : 전체 실험 사이클 (ENV + ALGO + validate + report)
- `/validate-code`    : 코드 정합성 체크 (A~D, 빠름)
- `/validate-method`  : 방법론 타당성 체크 (E~H, 심층)
- `/report`           : Notion 보고서 생성
- `/debug`            : 학습 결과 디버깅 (자동 WTW/AMP 분류)
- `/debug-wtw`        : WTW tracking + foot impact 전용 분석
- `/debug-amp`        : AMP discriminator + gait 다양성 전용 분석
- `/debug-proactive`  : 능동적 전체 학습 결과 이상징후 스캔
- `/research`         : Gemini 협업 deep research (논문 리서치 → 방향 제안 → 구현)
- `/research-team`    : 리서치 팀 (surveyor + advisor 토론) → 심층 방법론 검토 + 최종 제안
- `/locomotion-loop`  : 보행 학습 전체 루프 (ENV→ALGO→validate→학습→평가→피드백→반복)

## 전역 DO / DON'T

**DO:**
- 환경 작업 시 해당 디렉토리의 `*_context.md`를 먼저 읽어라
- config 변경은 반드시 `*_env_cfg.py`에서, 로직 변경은 `*_env.py`에서
- 새 텐서 추가 시 `_reset_idx`에서 반드시 초기화
- AMP 관련 작업 시 `rsl_rl/rsl_rl_algorithms.md`를 참조

**DON'T:**
- `source/isaaclab/` 내부 코어 파일 직접 수정 금지
- `python` 직접 실행 금지 — 반드시 `./isaaclab.sh -p` 사용
- 검증 없이 reward weight 합산이 설계 의도를 벗어나는 값 사용 금지
- 구현 전 `/validate` 없이 큰 변경 진행 금지

## 코드 스타일
- 라인 길이: 120자
- Linting: ruff (E, W, F, I, UP, C90, SIM, RET)
- Type check: pyright (basic mode)
- `@configclass` 데코레이터를 모든 config 클래스에 사용

## 학습 출력
- 로그/체크포인트: `logs/` 디렉토리
- WandB 권장: `--logger wandb --wandb-project IsaacLab-locomotion`
  (TensorBoard 대비: 브라우저 실시간 접근, 실험 간 자동 비교 가능)

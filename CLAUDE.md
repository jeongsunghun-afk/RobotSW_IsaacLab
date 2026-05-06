# IsaacLab Project Guide

## 프로젝트 미션
다양한 robot과 task에 대한 IsaacLab 시뮬레이션 환경을 만들고, 강화학습/모방학습으로 학습한 정책을 실제 로봇에 배포하는 것이 목표.

핵심 가치
- **강건성**: 명령(속도/목표/etc) 정확 추종, 외란/도메인 변화에 강인
- **자연스러움/품질**: 부드러운 동작, 안전한 접촉, sim-to-real 전이 가능성
- **확장성**: Go2 / R_Skeleton 같은 기존 로봇 외에도 새 robot/task를 빠르게 추가

지원 알고리즘 패러다임 (모두 1급 시민)
- 일반 강화학습 (PPO 계열, 향후 SAC/PPG 등)
- 모방학습 (BC, GAIL, AMP, Diffusion 등)
- 하이브리드 (RL + 모방학습 결합)

## 실행 환경
- Isaac Sim 기반 GPU 가속 RL 프레임워크
- Python 3.10+ (Isaac Sim 4.5: 3.10 / 5.0+: 3.11)
- PyTorch 2.7+ (CUDA 12.8)

## 주요 명령어

### skrl

```bash
# 학습
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task <Task-Name> --num_envs 4096 --algorithm <PPO|AMP|...>

# 평가
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
  --task <Task-Name> --num_envs 32 --algorithm <PPO|AMP|...>
```

### rsl_rl

```bash
# 학습
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task <Task-Name> --num_envs 4096

# 평가
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task <Task-Name> --num_envs 32
```

### 공통 유틸

```bash
./isaaclab.sh -f    # 코드 포맷 (ruff)
./isaaclab.sh -t    # 테스트
./isaaclab.sh -i    # 패키지 설치 (개발 모드)
```

> 두 프레임워크는 알고리즘 라이브러리의 차이. 환경 내부 구조는 각 환경 디렉토리의 `CLAUDE.md` 참조.

## 프로젝트 구조

```
source/isaaclab_tasks/isaaclab_tasks/direct/
├── <task_dir_1>/   → CLAUDE.md (있으면)
├── <task_dir_2>/   → CLAUDE.md
└── ...

skrl/               → CLAUDE.md  (커스텀 skrl 라이브러리)
rsl_rl/             → CLAUDE.md (있으면)
scripts/reinforcement_learning/
├── skrl/           → train.py, play.py, commands.yaml
└── rsl_rl/         → train.py, play.py
```

> 환경 추가/이름은 `source/isaaclab_tasks/isaaclab_tasks/__init__.py` 등록 참조.

---

## 핵심 원칙: 메인 Session은 Orchestration Only

> **메인 agent는 직접 코드를 작성·수정·탐색하지 않는다.**
> 모든 실질적인 작업은 반드시 sub-agent(coordinator/worker)에게 위임한다.

### 위임 규칙

| 상황 | 행동 |
|------|------|
| 코드 탐색 | `Agent(subagent_type="Explore")` 또는 `oh-my-claudecode:explore` |
| 구현 (env/cfg/reward 등) | 해당 worker에게 `Agent` tool로 dispatch |
| 큰 변경/리팩토링 | `oh-my-claudecode:executor` 또는 worker 조합 |
| 분석/디버깅 | `debug-worker` / `log-analyzer` / `motion-analyzer` |

---

## Auto-routing 규칙

사용자 요청 수신 시 아래 규칙으로 자동 분류하고 **즉시 sub-agent에게 dispatch**한다.

### 라우팅 테이블

| 분류 키워드 | 호출 대상 | 비고 |
|-----------|---------|------|
| ENV (reward, obs, 보상, 환경, cfg, 관절, 발/엔드이펙터, 리셋, command/goal) | `/env-coordinator` skill | 세부 worker 자동 dispatch |
| ALGO (network, loss, PPO, AMP, IL, discriminator, 하이퍼파라미터) | `/algo-coordinator` skill | 세부 worker 자동 dispatch |
| 디버그, 분석, 학습 결과 이상, 문제 원인 | `debug-worker` (Claude-only) | 환경/알고리즘 자동 분류 |
| 학습 루프, 이터레이션, 전체 파이프라인 | `/locomotion-loop` skill | 본 스킬은 보행 외 task에도 일반 적용 가능 |
| 보고, report, notion | `/report` skill | |

---

## Worker 레지스트리

모든 worker는 `.claude/agents/<name>.md`에 정의되며 `Agent(subagent_type="<name>", ...)`로 호출한다.

### 환경 계열
| Worker | 주요 책임 영역 | 핵심 불변 규칙 |
|--------|----------------|---------------|
| obs-worker | `*_env.py` 흐름 전반 — `_get_observations`, `_reset_idx`, command/goal 로직, buffer 관리 | obs_space 차원 = `_get_observations` cat 결과 크기, 신규 buffer는 `_reset_idx` 초기화 |
| reward-worker | `*_env.py`의 `_get_rewards()` + cfg `reward_scales` | penalty=음수, bonus=양수, 신규 buffer→reset 초기화 |
| cfg-worker | `*_env_cfg.py` 또는 `agents/*_cfg.{py,yaml}` | `@configclass` 유지, 타입 일치, 의존 파라미터 동기화 |

### 알고리즘 계열
| Worker | 주요 책임 영역 | 핵심 불변 규칙 |
|--------|----------------|---------------|
| network-worker | rsl_rl 모듈(`actor_critic.py`, `*_discriminator.py`) 또는 skrl YAML `models` | input/output shape 보존, forward 동기화 |
| loss-worker | rsl_rl/skrl 알고리즘의 `update()` | detach 위치, optimizer 분리, magnitude 균형 |
| hyperparam-worker | agent cfg(PPO/AMP/IL 하이퍼파라미터) | 한 번에 1~2개만 변경, 타입/범위 합리성 |

### 검증 계열
| Worker | 모델 | 역할 |
|--------|------|------|
| validate-code | haiku | 코드 정합성 (체크리스트 A~D) — observation/cfg/reward shape, IL 차원 동기화 |
| validate-method | sonnet | 방법론 타당성 (체크리스트 E~H) — RL/IL 이론 + sim-to-real, WebSearch 활용 |

### 분석 계열
| Worker | 모델 | 역할 |
|--------|------|------|
| log-analyzer | haiku | WandB/TF events/CSV 메트릭 추출 + Green/Yellow/Red 판정 |
| motion-analyzer | haiku | 모방학습 reference 데이터(motion/demo) 통계 + 커버리지 평가 |

### 디버그 계열
| Worker | 모델 | 역할 |
|--------|------|------|
| debug-worker | opus | 학습 실패/저성능의 root cause 분석. Claude-only(외부 CLI 의존 없음). 모든 task/algo. |

### 학습 루프 계열
| Worker | 모델 | 역할 |
|--------|------|------|
| training-evaluator | sonnet | 학습 결과 종합 평가 → `_workspace/eval_iter_{N}.md` |
| feedback-generator | sonnet | 평가 결과를 다음 이터레이션 작업 지시서로 변환 → `_workspace/feedback_iter_{N}.md` |

> 보고서 작성은 글로벌 agent `session-report-writer`(`~/.claude/agents/session-report-writer.md`) 및 `/report` 명령 사용.

---

## 역할 커맨드 (직접 호출용)
- `/env-coordinator`   : ENV 작업 분류 + 적절한 worker dispatch
- `/algo-coordinator`  : ALGO 작업 분류 + 적절한 worker dispatch
- `/locomotion-loop`   : 학습 전체 루프 (ENV→ALGO→validate→학습→평가→피드백→반복)
- `/report`            : Notion 보고서 생성

## 전역 DO / DON'T

**DO:**
- 환경 작업 시 해당 디렉토리의 `CLAUDE.md`(있으면)를 먼저 읽기
- config 변경은 `*_env_cfg.py` 또는 agent cfg에서, 로직 변경은 `*_env.py`에서
- 새 텐서 buffer 추가 시 `_reset_idx`에서 반드시 초기화
- 큰 설계 변경 시 `validate-method` 우선 실행

**DON'T:**
- `source/isaaclab/` 내부 코어 파일 직접 수정 금지
- `python` 직접 실행 금지 — 반드시 `./isaaclab.sh -p` 사용
- 한 번에 너무 많은 변경 동시 적용 금지 (원인 분리 불가)
- **메인 agent가 직접 Edit/Write/Grep으로 코드 수정 금지** — 반드시 worker에게 위임

## 코드 스타일
- 라인 길이: 120자
- Linting: ruff (E, W, F, I, UP, C90, SIM, RET)
- Type check: pyright (basic mode)
- `@configclass` 데코레이터를 모든 config 클래스에 사용

## 학습 출력
- 로그/체크포인트: `logs/` 디렉토리
- WandB 권장: `--logger wandb --wandb-project IsaacLab-<your-project>`
  (TensorBoard 대비: 브라우저 실시간 접근, 실험 간 자동 비교 가능)

# JSH — 내 산출물 폴더

RobotSW_IsaacLab에서 **내가 만든 새 산출물**을 이 폴더에 저장한다 (회사 원본 코드와 분리).

## GitHub 규칙
- `origin` = `github.com/jeongsunghun-afk/RobotSW_IsaacLab` = **내 개인 fork** → 이 폴더(JSH) push OK.
- `upstream` = `github.com/RGA-Robotics/RobotSW_IsaacLab` = **회사 repo** → **push 절대 금지** (no_push 설정됨).
- 회사 파일(rga.py 등) 수정도 origin에만 반영하고 upstream엔 올리지 않는다.

## 내용물
| 항목 | 설명 | git |
|---|---|---|
| `Hind_Leg/` | hind_leg.usd + configuration 레이어(66M). RL 로봇 자산. rga.py L438이 참조 | **제외**(`.gitignore:12 **/*.usd`) — 2026-10-07 실측: 추적 USD 0/10개. 새 머신에선 `HindLeg-Direct-v0` 생성 불가 |
| `Hind_Leg_Flat/` | 평발 2점 변형(65M). 충돌구 수편집 산물이라 **재생성 불가** | **제외**(동일 규칙) — 서버/별도 백업 필수 |
| `서버접속_가이드.md` | GPU 서버 SSH/학습 실행 가이드 | **제외**(비밀번호 포함, .gitignore) |
| `HANDOFF_2026-10-07_rpet_go2.md` | **★인수인계** — R.pet 정상화·험지 zero-shot·걸음분류 결과 + Go2 TAMOLS 검증 종결 + 다음 할 일. **§6 철회목록·§7 측정함정 먼저 읽기** | 커밋 |
| `handoff_2026-10-07/` | 위 문서의 실행 스크립트·패치·결과 원자료 사본(300K) | 커밋 |
| `isaac6_결론문서_정리.md` | isaac-6.0 라인 조사(2026-09-09). 인수인계 문서가 전제로 참조 | 커밋 |

## hind_leg USD
- `Hind_Leg/hind_leg.usd` = `HindLeg-Direct-v0` RL 태스크가 로드하는 로봇 USD.
- 원본은 lgb 개인머신(`/home/lgb/...`)에만 있어 여기로 복사함.
- **rga.py L438 usd_path를 이 폴더 경로로 수정해야 함**:
  - 서버: `/mnt/ssd1/jsh/RobotSW_IsaacLab/JSH/Hind_Leg/hind_leg.usd`

## 서버 학습 (요약)
```bash
ssh jsh@192.168.1.205
cd /mnt/ssd1/jsh/RobotSW_IsaacLab
CUDA_VISIBLE_DEVICES=2 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task HindLeg-Direct-v0 --num_envs 4096 --headless --logger wandb --wandb-project IsaacLab-HindLeg
```
> GPU 2번만 사용. 상세는 `서버접속_가이드.md`(로컬).

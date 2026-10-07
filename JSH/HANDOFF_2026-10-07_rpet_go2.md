# 인수인계 — R.pet 정상화 + Go2 TAMOLS 검증 (2026-10-07)

다른 머신/다른 세션에서 **바로 이어받아 실행**할 수 있도록 쓴 문서다.
읽는 순서: §0 접속 → §1 지금 상태 → §2 다음 할 일 → §3~ 세부.

> **이 문서의 가장 중요한 부분은 §6(철회·정정)과 §7(측정 함정)이다.**
> 이 라인에서 "결론처럼 보이던 것"이 측정·집계·조회 조건의 artifact 였던 사례가 **10건 이상**이다.
> 새 결과를 믿기 전에 §7 을 먼저 읽을 것.

---

## 0. 접속과 환경

### GPU 서버
```bash
ssh -i ~/.ssh/id_rga jsh@192.168.1.205   # 키 없으면 JSH/서버접속_가이드.md (로컬 전용, 커밋 제외)
```
- GPU 4장(RTX 4090 24.5 GB). 드라이버 570.172.08 / CUDA 12.8.
- **GPU2 만 쓴다** (사용자 지시). GPU0·1·3 은 다른 사용자(cky, root) 작업이 수시로 올라온다.
- 다른 사용자: `cky`(GPU0·1 상시), `/home/lgb`(이 R.pet 인계본의 송신자, 접근 불가)

### conda 환경 — **두 개를 절대 섞지 말 것**

| env | python | 용도 | 핵심 버전 |
|---|---|---|---|
| `isaac-5.1` | 3.11.15 | **Go2 작업 전용** | isaacsim 5.1.0.0 · isaaclab 0.54.3 · rsl-rl-lib 3.2.0 · torch 2.7.0+cu128 |
| `isaac6` | 3.12.13 | **R.pet 작업 전용** | isaacsim 6.0.1.0 · isaaclab 6.1.11 · rsl-rl-lib 5.0.1 · torch 2.10.0+cu128 |

`isaac6` 는 인계본 `env/VERSIONS.md` 명세와 **8/8 일치**하게 새로 구축했다.
`isaac-5.1` 은 손대지 않았다(Go2 작업 전체가 거기 걸려 있다).

> **★isaac6 의 CUDA 라이브러리 주의**: `pip install isaacsim[all,extscache]` 는 **cu13 계열 15개**를
> 끌어와 torch cu128 과 충돌해 `CUDNN_STATUS_NOT_INITIALIZED` 가 난다. 인계본 yml 은 **cu12 만** 쓴다.
> 현재 env 는 cu13 을 전부 제거하고 cu12 로 핀 고정해 둔 상태다(`cudnn 9.10.2`). **isaacsim 을
> 재설치하면 이 충돌이 되살아난다** — 재설치 후엔 반드시 `nvidia-*` 목록에서 cu13 을 제거할 것.

### 저장소

| 경로 | 내용 | remote |
|---|---|---|
| `/home/jsh/문서/jsh` | 로컬 작업 루트. `simulation/docs/` 에 개발리포트 | **없음**(푸시 불가) |
| `/home/jsh/문서/jsh/RobotSW_IsaacLab` | 브랜치 `feat/quad17-dtc-gaptest`. **이 문서가 있는 곳** | origin = `github.com/jeongsunghun-afk/RobotSW_IsaacLab.git` |
| `/home/jsh/문서/jsh/RobotSW_IsaacLab_isaac6` | 워크트리. **실측 브랜치 = `jsh/quad17-dtc`**(a2f547d09) — 이름은 isaac6 지만 내용은 그게 아니다. **이 브랜치가 미푸시 parkour 커밋 2개를 들고 있다 — 지우지 말 것** | 같은 저장소 |
| 서버 `/mnt/ssd1/jsh/RobotSW_IsaacLab` | Go2 작업 트리(2.3.2) | — |
| 서버 `/mnt/ssd1/jsh/RobotSW_IsaacLab_isaac6` | R.pet 작업 트리(3.0.0) | — |
| 서버 `/mnt/ssd1/jsh/rpet_handoff/leg_locomotion_handoff_20260922` | R.pet 인계 패키지 | — |

> **★서버 작업트리에서 `git` 이 동작하지 않는다.** 서버의 두 트리는 로컬에서 rsync 된 것이라
> `.git` 파일이 **로컬 머신 경로**(`/home/jsh/문서/jsh/RobotSW_IsaacLab/.git/worktrees/...`)를 가리킨다.
> `git diff` 가 **에러 없이 빈 출력**을 내므로 "변경 없음"으로 오독하기 쉽다(실제로 겪었다).
> 서버 쪽 변경은 `diff -u <원본> <현재>` 로 떠야 한다. 내 R.pet 수정분은 이미 떠서
> `JSH/handoff_2026-10-07/rpet/rpet_my_edits.diff` 로 넣어 뒀다.
>
> **Go2 작업트리는 커밋되지 않은 변경이 5046줄 있다**(`go2/WORKTREE_diffstat.txt`).
> 이 중 `go2_wtw_env.py` 4246줄은 DTC 프로젝트 전체 누적분이지 이 세션 분이 아니다.
> **서버 트리를 reset/clean 하지 말 것 — 되돌릴 수 없다.**

### ★원격에 **없는** 것 — 새 머신에서 clone 하면 빠지는 것

`git clone origin` 으로는 아래가 **오지 않는다**(실측, 2026-10-07).

| 항목 | 기제 | 크기 | 작업 차단? |
|---|---|---|---|
| **`JSH/Hind_Leg*/` USD 10개** | `.gitignore:12 **/*.usd` | **130 MB** | **★예.** `rga.py:440/504` 가 이 경로를 참조 → `HindLeg-Direct-v0`·`HindLeg-Flat-Direct-v0` 로봇 생성 불가. **`Hind_Leg_Flat` 은 평발 2점 보행(RL 성공 결과)의 자산이고 충돌구 수편집으로 만든 것이라 재생성 불가** |
| **`jsh/quad17-dtc` 브랜치 커밋 2개** | 원격 추적 없음 | **고유 내용은 ~110줄**(7,800줄은 측정 착오 — 대부분 origin 에 이미 있는 트리의 재배치) | **★예.** `feat/quad17-dtc-gaptest` 푸시로 **커버되지 않는다**. origin 전 ref 에 없는 blob 5개: `rga.py`(37,514 B, **`LEG_DTC_CFG`=발목 토크 142.8→100.8 실측복원**) · `quad17_env.py` · `quad17_env_cfg.py` · `quad17_parkour/{__init__,parkour_env_cfg}.py` |
| **이 인계 커밋 2개** | 미푸시 | 57파일 | 예 — 푸시 전엔 번들 전체가 로컬에만 |
| `JSH/*.mp4` 4개 | `*.mp4` | 3.7 MB | 아니오 — 같은 이름 `.gif` 가 커밋돼 있음 |
| `__pycache__/*.pyc` 7개 · `__MACOSX/` · `pace_hindleg/gt.pt` | 각 규칙 | 85 KB | 아니오 — 재생성됨 |
| `JSH/서버접속_가이드.md` | 의도적 제외(비밀번호) | 3 KB | **git 밖에 따로 백업할 것** — 서버 접속정보의 유일한 사본 |

**★`upstream/isaac-6.0` 브랜치는 서버에서 사라졌다.** `git ls-remote --heads upstream` 실측 결과
`isaac_5.1 · lgb/{imitation,parkour-pure-rl,real2sim,stage1-skill-tokenizer} · main` 뿐이다.
로컬 `refs/remotes/upstream/isaac-6.0` 은 **낡은 캐시**이고 `git fetch upstream` 으로는 다시 못 가져온다.
명시 SHA 로는 아직 fetch 된다 — **이 값을 이 디스크 밖에 적어 둘 것:**

```
isaac-6.0 = bea85978dd4f59c8f260217bdfcff49cb1629a32
git fetch upstream bea85978dd4f59c8f260217bdfcff49cb1629a32   # SHA 로는 성공 확인(2026-10-07)
```

**★origin/main 은 HEAD 의 선형 조상이다**(`merge-base --is-ancestor` = yes, `0 6`).
따라서 `git clone` 후 `git checkout feat/quad17-dtc-gaptest` 하면 **푸시된 quad17/DTC 커밋 4개는 전부 온다.**
빠지는 건 미푸시 2개뿐이다("clone 하면 main 이라 DTC 작업이 없다"는 설명은 틀렸다).

**★git-lfs 함정**: 추적된 **55개 파일(.mp4 39 + .obj 16)이 LFS 포인터**인데 이 머신엔 **git-lfs 가 없다**
→ Go2 메쉬(`source/isaaclab_assets/data/Robots/go2/meshes/*.obj`)가 **132바이트 stub** 이다.
**그리고 origin 의 LFS 저장소에 객체가 없다** — batch API 직접 조회 결과 `{"code":404,"message":"Object does
not exist on the server"}`(3개 표본). 즉 **git-lfs 를 깔아도 origin 에서는 복구 불가**다. upstream 은 401(비공개)
이라 RGA-Robotics 쪽엔 남아 있을 수 있다. 푸시로 해결되지 않는다.

**★`Isaaclab_Parkour`** = `.gitmodules` 없는 gitlink(mode 160000 → d6766d88). 로컬도 **빈 디렉터리**라
잃을 건 없지만 `git submodule update --init` 은 영구히 실패한다(URL 이 어디에도 없음).

### ★이 저장소 밖의 로컬 전용 자산

| 항목 | 상태 |
|---|---|
| `/home/jsh/문서/jsh/simulation/` | **자체 git repo** — origin=`jeongsunghun-afk/simulation`, 브랜치 8개 전부 푸시됨(안전). **단 미커밋 변경 872줄/14파일**이 로컬에만 — 특히 `docs/DTC_개발리포트.md` +104줄(P4 Go2 gradient-TO). **원격 사본은 구버전인데 완전해 보인다** |
| `/home/jsh/문서/jsh` (외곽 repo) | **remote 없음.** HEAD 에 추적파일 0개 · 커밋 14개(내용은 `simulation/docs/RUN.md` 로 이전돼 중복) |
| `RobotTestGait/` | 로컬 전용 **1,303줄** 임베디드 모터/게이트 bring-up C++. `defineConfigMotor.h` = 축별 채널·기어·sign/offset |
| `HANDOFF_claudeA_RL.md` · `면접준비_RL파이프라인.md` | 로컬 전용 작업문서 |
| `~/.claude/projects/-home-jsh----jsh/memory/` | **40파일 1 MB.** 프로젝트 문서가 `[[wikilink]]` 로 참조하는데 git 밖에 있다 |
| `02_Leg_UFDF_*` 4개 · `robot_viewer/_mymodels/` | 206M + 41M 원본 CAD/STL. 가공 메쉬(`quad/meshes_sim_17dof`)는 커밋돼 있어 MJCF 로드는 작동 |
| 서버 `/mnt/ssd1/jsh/RobotSW_IsaacLab` | 미커밋 ~34파일(walk 게이트·stepping 지형). **로컬 체크아웃만 보면 "walk 게이트 없음"으로 오진한다**(실제로 한 번 발생) |

관련 기존 문서(반드시 읽을 것):
- `JSH/isaac6_결론문서_정리.md` — **2026-09-09 사용자 직접 조사 242줄.** isaac-6.0 라인 전체 정리,
  R.pet/hind_leg/real2sim 판정, §F 이주 결정, §G 이주 조사(G-2 import 갭 없음 / **G-4 거동 리스크**).
  나는 이 문서를 읽지 않고 이미 조사된 것을 중복 측정한 적이 있다. **먼저 읽을 것.**
- `/home/jsh/문서/jsh/simulation/docs/DTC_개발리포트.md` **§P3.0** — Go2 4팔 통제실험 + 철회절.

---

## 1. 지금 상태 — 확정된 것

### 1-A. R.pet 정상화 — **완료**

| 단계 | 결과 |
|---|---|
| 환경 | `isaac6`, 인계본 VERSIONS.md **8/8 일치** |
| 모델 로드 | `strict=True` 통과 (actor 83→512→256→128→17, critic 101→…→1) |
| sim 구동 | 평지 vx1.0 추종 **97.2~98.8%** · 낙상 0 · estimator 오차 0.026 m/s |
| 관측·행동 정합 | `test_parity --with_env` **(a)(b) PASS**, max diff **3.1e-06** (기준 1e-4) |

**원본 `play_in_isaaclab.py` 가 수정 없이 돈다.**

### 1-B. R.pet 험지 zero-shot (재학습 0) — **완료**

vx 1.0 고정 · yaw **0 고정** · 16 env · 400 step:

| 지형 | L | vx실측 | 추종률 | 이동거리 | 플랫폼이탈 | 낙상 env |
|---|---|---|---|---|---|---|
| 평지 | — | 0.988 | 98.8% | 7.93 m | 100% | 0% |
| rough | 2 | 0.952 | 95.2% | 7.68 | 100% | 0% |
| rough | 5 | 0.921 | 92.1% | 7.36 | 100% | 0% |
| rough | 8 | 0.733 | 73.3% | 3.12 | 81% | 81% |
| gap | 2 | 0.981 | 98.1% | 6.65 | 94% | 25% |
| gap | 5 | 0.979 | 97.9% | 6.28 | 94% | 31% |
| gap | 8 | 0.991 | 99.1% | 5.23 | 88% | 56% |
| stepping | 2 | 0.978 | 97.8% | 7.90 | 100% | 0% |
| stepping | 5 | 0.949 | 94.9% | **0.50** | **12%** | **100%** |
| stepping | 8 | 0.943 | 94.3% | **0.09** | **0%** | **100%** |

> **추종률만 보면 안 된다**: stepping L8 은 추종률 94.3% 인데 이동거리 **9 cm** 다.
> 제자리에서 발을 놀리며 몸통 속도만 맞춘다. 반드시 `far_mean`(이동거리)과 함께 읽을 것.

### 1-C. R.pet 걸음 분류 — **완료(저속 한정)**

vx 1.0 · 64 env · 마지막 2 s 창:

| 지형 | L | pace | trot | bound/gallop | 적합도 | 낙상env |
|---|---|---|---|---|---|---|
| 평지 | — | **85.9%** | 14.1% | 0% | 0.246 | 0% |
| rough | 5 | 79.7% | 20.3% | 0% | 0.238 | 5% |
| rough | 8 | 71.9% | 15.6% | 12.5% | 0.259 | 66% |
| gap | 8 | 75.0% | 18.8% | 6.2% | 0.245 | 39% |
| stepping | 5 | 54.7% | 26.6% | 18.8% | 0.255 | 62% |
| stepping | 8 | 64.1% | 23.4% | 12.5% | 0.250 | 61% |

**험지에서도 pace 가 55~88% 로 지배적이고 적합도가 평지(0.246)와 동급이다** — 걸음이 무너져
"기타"가 되는 게 아니라 **pace 틀을 유지**한다.

### 1-D. 세 갈래 판정 — 모두 갈렸다

```
① 게이트    험지에서도 pace 유지 (평지와 같은 적합도)
             → TAMOLS/GIAC 는 지지폴리곤 전제. pace 는 지지선이 몸통 한쪽 전후선이라
               공칭 CoM 투영을 지나지 않는다. **접합면이 없다.**
② rough/gap  지각 없이 상당 부분 통과 (rough L5 낙상 0 · gap 추종 98~99%)
             → TAMOLS 고유 이득 기대 0. Go2 P2.7·P4 와 일치.
③ stepping   blind 완전 실패 (L5+ 이동 0.09~0.50 m · 전 env 낙상)
             → **참조 슬롯 필요가 R.pet 에서 확정.** Go2 (d) 조건 재현.
```

**권고(유지)**: 안 C = **지각(height scan) 추가 + 험지 fine-tune, TAMOLS 캐시는 붙이지 않음.**
참조 슬롯은 **온라인 Raibert + 최근접돌 스냅**(env 안 50줄)으로 채운다.
근거: ③이 유일한 투자 근거인데 "참조 슬롯이 필요"까지만 말하고, Go2 실측이 싼 것과 TAMOLS 가
동률(난도지형에선 TAMOLS 열세)이라고 말한다. ①이 가장 무겁다 — TAMOLS 를 쓰려면 게이트를
바꿔야 하고, 그건 AMP 가 보상 50% 로 붙든 prior 와 싸우는 **재학습**이며 이 체크포인트의 유일한
자산(고속 추종 97~98%)을 버리는 일이다.

### 1-E. Go2 TAMOLS 검증 — **판정 불가로 종결(설계 재정비 필요)**

팔 구성(전부 징검돌·walk 고정게이트·seed 42 기준):

| 팔 | 발판 참조원 | 몸통참조 | L9 CROSS/24 (seed 42,7,1,2,3) |
|---|---|---|---|
| **G** | 기하 Raibert+최근접돌 스냅 | critic 전용 +6 | 7, 2, 0, 15, 17 |
| **N** | 기하 스냅 | 없음 | 0, 0, 13, 0, 0 |
| **P** | 기하 스냅 | 정책관측+보상 | 0, 14, 12, 13, 6 |
| TS | TAMOLS 캐시 + 스냅 | critic | (seed42) 1 |
| T | TAMOLS, **스냅 없음** | critic | ~~무효~~ (스냅이 제2변수) |
| S′ | 기하 스냅 | 캐시 **위상평균** | (seed42) 0 |

**어느 쌍도 유의하지 않다**(MWU 양측 P-N 0.167 · P-G 0.897 · G-N 0.127~0.15).
그리고 **n=5 는 원리적으로 판정 불가**: 비대응 MWU 양측 최소 달성가능 p = 2/252 = **0.0079**,
p≤0.05 는 U≤2 = 25개 쌍비교 중 역전 2개 이하 = **사실상 완전분리** 필요. 관측 산포(0 동점 포함)
에서 비현실적. n=8 → U≤13, n=10 → U≤23.

---

## 2. 다음에 할 일 (우선순위)

### 2-1. R.pet — 안 C 착수 (권고 경로)
1. **지각 관측 추가** — `obs_groups` 에 `"scan"` 키 한 줄 → `ActorCriticRMA` 가
   `scandot_encoder` MLP[128,64,32] 를 자동 생성하고 latent 32 를 **actor 입력 맨 끝**에 concat한다
   (`env/rsl_rl/rsl_rl/modules/actor_critic_parkour.py:127-131, 148-150, 173-178`).
   끝 삽입이라 32열 0-init 로 기존 정책 출력이 보존된다. **critic 에도 넣을 것**(지형 배정이 정책
   선택 밖이라 critic 이 지형별 baseline 을 배워야 advantage 가 오염되지 않는다 — 이 판단은 추론).
2. **지형 env** — 이미 `RPET_TERRAIN` 게이트를 넣어뒀다(§4). 종료/스폰의 절대 world z 를
   **지형 상대로** 바꾸는 것이 남았다(`leg_imitation_tracking_env.py` 종료 :405 부근, RSI 스폰).
3. **참조 슬롯** = 온라인 Raibert + 최근접돌 스냅. 캐시·격자·솔버 포팅 비용 0.
4. **AMP** — 험지 env 만 `flat_mask=0` 으로 끈다(`leg_imitation_tracking_rma_env.py:51`, 1줄).
   그 순간 남는 보상이 속도추종 2항뿐이므로 **발 클리어런스·자세·높이 항을 신설**해야 한다.
5. 3~5 시드 × DTC-OFF 기준선. `terrain_level` **과 렌더**를 함께 볼 것.

> **★무음 실패 함정 3개(반드시 피할 것)**
> ① 참조를 기존 `priv`(38) 그룹에 넣지 말 것 — priv 는 critic 전용이 **아니다**. 롤아웃 20 iter 중
>    19 를 actor 가 priv latent 로 돌고 history_encoder 가 그걸 DAgger 로 재현하도록 학습된다
>    (`actor_critic_parkour.py:317-326`) → 학습곡선은 좋고 **배포 경로(`act_inference`)만 무너진다.**
> ② **symmetry data-aug 가 기본 ON** 이고 모르는 그룹을 **미러 없이 복제**한다
>    (`mdp/symmetry.py:212` `obs.repeat(2)` 후 4키만 미러) → 배치 절반이 "몸은 좌우반전·지형은 원본"
>    모순 샘플, 에러 0. **scan 미러 함수(좌우 스왑 + y 부호반전) 필수.**
> ③ **scan 은 추론 경로에서 정규화되지 않는다**(`scan_obs_normalizer` 가 `act`/`act_inference` 에서
>    미사용) → heightmap 을 직접 클리핑·스케일해 넣어야 한다.

### 2-2. R.pet — 미측정 보완
- **고속(vx 2~3) 걸음 분류** — 현 분류기는 저속만 검증됨(§3-C). 2 s 창 FFT 분해능 0.5 Hz 가
  한계로 보인다. 창을 4 s 로 늘리거나 Hilbert/제로크로싱 기반으로 바꿀 것.
- **접촉 스케줄 역추정** — TAMOLS 를 끝까지 보려면 필요. 접촉 센서가 죽어 있으니(§5) 발 높이·속도로
  대체하거나 진단 경로에서만 센서를 고칠 것.

### 2-3. Go2 — 설계 재정비 후 재실행
**0단계(선행, GPU 0h)** — 집계 툴체인 고정:
- `rs_table_arms.py:7` `SEEDS` 기본값 제거 → 미지정 시 디렉터리 전체 사용 + 사용 seed 를 헤더에 인쇄
- `:21` 무경고 `continue` → stderr 경고
- **R4 CEILING 섹션 `n=8` 하드코딩 수정** (현재 리포트가 문자 그대로 `23/8` 을 인쇄하고
  PASS 기준 `cr>=7` 이 n=24 에서 87.5% 가 아니라 **29%** 다)
- `robust_suite_arms.sh` 의 table/report 모드가 구버전 `/tmp/rs_table.py`(arm 정책 0/10 매칭)를
  호출하는 것 수정 · 공유 전역경로 `/tmp/rs_rows.json` 폐기

**1단계 — 헤딩 구멍 선행 수정**: world yaw/progress 항 추가 또는 지속적 헤딩 반전 종료.
`GO2_CORRIDOR_DONE=1` 로는 안 된다(`go2_wtw_env.py:1721` 이 "it does not touch this defect" 라 명시).
**정확성 문제가 아니라 검정력 문제다** — 1/5 seed 가 terrain_level≈0 바닥에 박히면 동점으로
MWU 검정력이 날아간다. 이걸 하면 **기존 G 5런은 대조군 재사용 불가**(전부 재실행).

**2단계 — 본 실험**: 팔 **G vs G+R**(동일 + `base_ref_track` 보상만).
- 보상 게이트 분리 필요: 현재 `:4654` 가 `_sref_base_policy` 에 묶여 "보상만" 팔이 없다
  → 독립 env 플래그 신설 ~3줄
- **scale ≈ 0.13** 1차 투여(아래 §6-E 참조), **σ=0.05 고정**
- seed 각 팔 **10개**, 주지표 **하나**(최종 100 iter 평균 `Metrics/terrain_level`),
  검정 **하나**(비대응 MWU 양측 α=0.05). **대응검정 금지**(쌍 상관 음수)
- 비용: 3.4~3.6 h/런 × 20런 = **70 GPU-h**

> **솔직한 경고**: 순항 구간 오차에서 x 가 60~72%(전진 지체는 `track_lin_vel` 이 이미 담당),
> z 가 18~31%(`base_height` 가 부분 담당), **y 만 10%** 다. 이 항의 고유 정보는 횡방향 10% 수준이고
> 참조의 지형인식 성분도 RMS 0.017~0.044 m 로 커널폭(0.224 m)의 1/5~1/13 이다.
> **용량을 고쳐도 null 이 나올 가능성이 높고, 그걸 받아들일 각오로 사전등록해야 한다.**

---

## 3. 실행 방법 (복붙 가능)

### 3-A. R.pet 평지 play (정상화 확인)
```bash
ssh -i ~/.ssh/id_rga jsh@192.168.1.205 'bash /mnt/ssd1/jsh/rpet_play6.sh'
# 기대: 추종 97~99% · termination 0 · estimator 오차 ~0.026 m/s
```

### 3-B. R.pet 정합 판정
```bash
ssh -i ~/.ssh/id_rga jsh@192.168.1.205 'bash /mnt/ssd1/jsh/rpet_parity6.sh'
# 기대: (a) PASS · (b) PASS, max abs diff ~3e-06
```

### 3-C. R.pet 험지 zero-shot (교정판 — 이동거리 포함)
```bash
ssh -i ~/.ssh/id_rga jsh@192.168.1.205 'bash /mnt/ssd1/jsh/rpet_zs2.sh'
ssh -i ~/.ssh/id_rga jsh@192.168.1.205 'python3 /mnt/ssd1/jsh/tab_zs2.py'    # 표로 출력
```

### 3-D. R.pet 걸음 분류
```bash
ssh -i ~/.ssh/id_rga jsh@192.168.1.205 'bash /mnt/ssd1/jsh/rpet_gait.sh'          # 평지 검증(vx 1.0, 3.0)
ssh -i ~/.ssh/id_rga jsh@192.168.1.205 'bash /mnt/ssd1/jsh/rpet_gait_terrain.sh'  # 험지 전체
ssh -i ~/.ssh/id_rga jsh@192.168.1.205 'python3 /mnt/ssd1/jsh/tab_gait.py'
```
**검증 규약**: 평지 vx1.0 에서 pace 70~86% 가 나와야 분류기를 신뢰한다(문서값 70~71%).
고속은 검증 실패 상태 — vx 2~3 결과는 쓰지 말 것.

### 3-E. Go2 작업 큐
```bash
# 상주 러너(폴링형). 큐에 50_xxx.sh 를 넣으면 이름순으로 집어간다.
ssh ... 'ls /mnt/ssd1/jsh/gpu2q/'                        # 대기 작업
ssh ... 'tail -20 /mnt/ssd1/jsh/gpu2_queue_status.txt'   # 진행 기록
ssh ... 'setsid nohup bash /mnt/ssd1/jsh/gpu2_runner.sh > /tmp/r.log 2>&1 &'  # 러너 기동
# 정지: /mnt/ssd1/jsh/gpu2q/STOP 파일 생성
```
각 작업 스크립트는 **GPU2 여유 가드**(2 GB 미만까지 대기)를 넣어 cky 님 작업과 경합하지 않게 한다.

---

## 4. 내가 건드린 파일 — 전부, 이유와 함께

모두 **게이트·폴백·추가** 형태라 **미설정 시 원본 동작이 불변**이다(평지 회귀로 소수점까지 확인).

### R.pet (`/mnt/ssd1/jsh/RobotSW_IsaacLab_isaac6/`)

| 파일 | 변경 | 이유 | 백업 |
|---|---|---|---|
| `.../robots/rga.py` | `LEG_CFG.usd_path` 를 `LEG_ASSET_ROOT` 아래에서 찾게 | 원본은 `/home/lgb/...` 절대경로(repo 에 USD 없음, 09-09 §F-1) | — |
| `.../leg_imitation_tracking/*.py` | **인계본(9/22)으로 교체** | 브랜치(커밋 `bea85978d`, 9/7)에 `last_action_obs` 가 없다. 인계본이 체크포인트와 짝 | `leg_imitation_tracking.bak_branch_0907/` |
| `..._env_cfg.py` | `MOTION_FILES_DIR` 폴백 추가 | 기본값 `merged_leg_pkl` 이 repo 에 없다. 동봉 `new_smr_leg_pkl`(14 클립)로 폴백. play 는 env.yaml 로 덮으므로 영향 없고 **레지스트리 기본 cfg 경로(test_parity --with_env)만** 이 값을 쓴다 | — |
| `..._env.py` | `RPET_TERRAIN` 게이트 추가(`_setup_scene`) | cfg 의 `terrain` 필드는 **죽은 설정**이고 이 함수가 `spawn_ground_plane` 을 직접 부른다. 값: `rough\|gap\|stepping` + `RPET_TERRAIN_LEVEL=0..9` | `/mnt/ssd1/jsh/leg_env.bak_terrain` |
| `..._env_cfg.py` | contact `prim_path` 수정 **시도 → 원복** | §5 참조. 고치지 않는 것이 맞다 | `/mnt/ssd1/jsh/leg_cfg.bak_contact` |

추가로 만든 자산(사용하지 않지만 보존): `assets/usd_51/Leg_51.usd` — 같은 URDF 를 **5.1 변환기로
재임포트**한 USD. 6.0 USD 가 5.1 에서 `CreateJoint - no bodies defined` 를 내던 때 만든 것.
isaac6 로 가면서 불필요해졌으나 "5.1 에서도 돌 수 있다"는 사실의 증거로 남긴다.
`rga_leg.py` 의 `RPET_USD_PATH` 훅(서버 5.1 트리)도 같은 맥락.

### Go2 (`/mnt/ssd1/jsh/RobotSW_IsaacLab/.../direct/go2/`)
전부 env 게이트 추가(미설정 시 불변). `go2_wtw_env.py` 에:
- `GO2_STEP_TAMOLS_FH_SNAP` — TAMOLS 발판 xy 에 기하경로와 동일한 최근접-돌 스냅 적용
- `GO2_SREF_BASE_ANALYTIC` — 몸통참조를 캐시 **위상평균**으로(지형 조건화만 제거) + `BASEREF_SRC` 배너
집계 측: `rs_table_arms.py` 정규식 `(d\d+|arm[A-Za-z0-9]+)` · `SEEDS` 를 `RS_SEEDS` 로 인자화
스위트: `robust_suite_arms.sh` 에 `armG/T/TS/N/S/S2/G7/S27/Gs*/Ns*/Ps*` case 추가

### 내가 만든 스크립트 — **이 커밋에 사본 있음**

사본: `JSH/handoff_2026-10-07/{rpet,go2,patches,results}/` (원본은 서버 `/mnt/ssd1/jsh/`)
```
rpet_zeroshot2.py   험지 zero-shot 계측기(★이동거리 포함 — 교정판)
rpet_gait.py        걸음 분류기(쌍 구조 기반)
rpet_contact_check.py  접촉센서 검증
rpet_play_2x.py     IsaacLab 2.3.2 런처 어댑터(isaac6 로 가며 불필요해졌으나 보존)
usd_probe.py        USD 강체 prim 구조 조사
tab_zs2.py / tab_gait.py   결과 표 출력
rpet_{play6,parity6,zs2,gait,gait_terrain,contact,convert}.sh   실행 래퍼
train_arm_{G,T,TS,N,S,S27,G7,P,generic}.sh   Go2 팔 학습
gpu2_runner.sh      GPU2 직렬 큐(상주형)
patch_*.py          위 수정들을 재현하는 패치 스크립트(멱등 아님 — 한 번만 적용)
```

### 결과 데이터 (서버)
```
/mnt/ssd1/jsh/armeval   1.2G  Go2 1차 평가(팔G/N/TS/S)
/mnt/ssd1/jsh/armeval2  3.5G  Go2 다중seed·팔P 평가 + REPORT_*.md
/mnt/ssd1/jsh/armeval3  2.3M  팔P base참조 덤프
/mnt/ssd1/jsh/rpet_zs2.log · rpet_gait_terrain.log   R.pet 험지 결과 원자료
/mnt/ssd1/jsh/armeval/p3_baseref.txt  1.07GB  ★분석 끝남. 샘플(p3_baseref_sample.txt)만 남기고 지워도 된다
```

---

## 5. 접촉 센서 — 죽어 있다. 고치지 마라(조건부)

**실측**: 추적 body **1개(Base)**, 120 스텝 접촉력 **전부 0.00 N**(영-액션으로 주저앉았는데도).
**원인**: `prim_path="/World/envs/env_.*/Robot/.*"` 가 **한 단계만** 매칭하는데 이 USD 의 강체는
운동학 사슬대로 중첩돼 있다 — `<root>/Geometry/Base/HL_hip_link/.../HL_foot_contact_link`, 깊이 3~9.
강체 22개(발 관련 8개)가 더 깊이 있어 안 잡힌다.

**그런데 이것은 원래 상태다.** `leg_imitation_tracking_env_cfg.py` 의
`contact_force_threshold: float = 500.0  # base 접촉 판정 [N] (contact sensor 결함으로 현재 사실상 무력)`
주석이 **브랜치와 인계본 양쪽에** 있다. 09-09 §G-4 도 경고했다.

**고치지 않는 이유**: ①`_get_dones` 의 **높이(0.35 m)·총기울기**가 접촉과 무관하게 작동해 낙상을
포괄한다(그 주석이 "축별 roll/pitch 는 roll≈49° 반쯤 누운 자세가 새는 사각지대가 있었다"며 교체했다
— hind_leg 의 "에피소드 92.9% 를 누운 채" 결함을 이미 고친 흔적) ②정책 obs 에 접촉이 없다
③걸음 분류도 thigh 위상 기반이라 접촉 불요 ④고치면 **학습 조건에서 벗어난다**.

**단 TAMOLS 접촉 스케줄 역추정에는 필요하다** — 그 단계에서 **진단 경로에만** 켤 것.
정규식 수정 시 주의: IsaacLab 경로 매처는 `/` 로 토큰 분해해 토큰별 컴파일하므로
`(/.*)*` 같은 그룹은 `unbalanced parenthesis` 로 거부된다.

---

## 6. ★철회·정정 목록 — 같은 실수를 반복하지 않기 위해

### 6-A. Isaac Sim 6.0 가용성 (3단 연쇄 오류)
- 잘못: "isaacsim 6.0 은 어느 pip 인덱스에도 없다" → "환경 바이너리 재구성 불가" → "lgb 님께
  `conda-pack` 을 요청해야 한다"
- 사실: **5.1 은 cp311 휠, 6.0+ 는 cp312 휠.** py3.11 환경에서 `pip index versions` 를 돌려
  cp311 휠이 있는 버전만 보였다. `--python-version 3.12` 로 조회하면 `6.0.0.0 … 6.1.0.0` 이 나온다.
- **교훈: 도구 출력을 무엇이 제약하는지 확인하지 않고 읽었다.**

### 6-B. upstream 브랜치 수
- 잘못: "`upstream` 에 브랜치가 1개뿐이라 isaac-6.0 소스가 없다"
- 사실: `git ls-remote --heads upstream | wc -l` 이 센 1 은 **인증 실패 에러 1줄**이었다.
  `upstream/isaac-6.0` 는 이미 fetch 돼 있고 워크트리까지 있었다.

### 6-C. 이식 표류 측정 중복
- 09-09 문서 §G-2 가 "IsaacLab 2.3.2 → 3.0.0 import 갭 없음"을 이미 확인해 뒀는데 읽지 않고
  반대 방향을 다시 측정했다. **기존 문서를 먼저 읽을 것.**

### 6-D. base 참조 오차 39 cm
- 잘못: 로그값 0.0098 에서 역산해 "오차 39 cm 라 σ 가 타이트해 보상이 죽었다"
- 사실: 실측 **5.3~6.5 cm**, `exp(−e²/0.05) ≈ 0.90`(거의 포화). 역산의 전제가 틀렸다.
- 살아남은 실제 신호: 오차가 **게이트 위상에 따라 3→8 cm 단조 증가**하고 증가분이 거의 전부
  전후 성분 = **참조가 정책이 내는 실제 전진보다 앞서 나간다.**

### 6-E. 보상 scale 배수
- 잘못: "scale 이 9.4배 작다 → 0.117 이어야 한다"
- 사실: 로깅이 `episodic_sum_avg / max_episode_length_s`(`go2_wtw_env.py:4863`)이고 분모는
  **고정 상수 30.0 s**. 주석의 산출식(`:4663`)이 16.4 s 를 한 번 더 나눠 **정확히 30배** 과소.
  의도 지분(양버킷 10%)을 내는 값은 **0.23**(실측 duty 0.41~0.93 반영), foothold 와 실제 동급은
  **0.13**. **σ=0.05 는 주석이 유일하게 맞게 고른 값 — 건드리지 말 것.**
- 같은 버그는 `base_ref_track` **단 하나**. D11/D16/P3/P4 캘리브레이션은 전부 **무차원 비**라 면역.
- **횡단 이슈**: 분모가 고정 30 s 이므로 **런 간 `Episode_Reward` 절대값 비교는 생존률(duty) 가중**
  이다(실측 duty 0.41~0.93, 2.3배 폭). 같은 런 안의 지분 비교는 안전.

### 6-F. 팔 P 해석
- 잘못: "강한 채널도 효과 없음(p=0.897)" → 이후 "사실상 관측만(+6)"
- 사실: 보상은 설계의 **1/19**(지분 0.43~0.55% vs 목표 10%) = **실제 처치 미시행**.
  관측은 +6 이 아니라 **+126 차원**(`observation_space += 6×(1+history_len)`, 20스텝 히스토리).
  그리고 **P ⊃ G**(critic +6 유지 + policy +126 + 보상) = 한 변수 대조가 아닌 복합 처치.
  seed42 붕괴는 P 의 처치효과가 **아니다** — critic 전용 `armS` seed42 도 같은 **180° 헤딩 뒤집힘**
  으로 붕괴했고 seed42 는 G·N 에서는 정상이다(env 결함: 보상이 base 프레임 속도추종이고
  world 헤딩 항이 없다).
- **p=0.897 은 "효과 없음"이 아니라 "측정 없음"이다.**

### 6-G. 팔 비교 결론 (철회)
- "L8 n=24 완전동률 → TAMOLS 발판 이득 없음"을 세 번째 replication 으로 보고했으나,
  **학습 seed 를 바꾸면 부호가 뒤집힌다**(L9 진입 G 19/24 vs S′ 5/24, p=0.00012 →
  G7 7/24 vs S27 18/24, p=0.0034 **반대방향**. 합산 p=0.68).
- **평가 24 seed 는 롤아웃 seed 이지 학습 seed 가 아니다.** 정책 자체는 팔당 n=1 이었다.

### 6-H. "양자택일" 오진
- 잘못: "L9 는 뚫거나 못 뚫는 이산 모드"
- 사실: **연속 단조 + 바닥 절단.** Spearman(최종 terrain_level, L9 CROSS) = **+0.937**(p=6.2e-5, n=10런).
  0/24 런들은 커리큘럼이 3000 iter 안에 L9 근처에 못 간 런. **고정 iter 예산이 10런 전부를 등반
  중 절단**(전 런 level 꼬리 상승 중, 상한 9 미달) = 검열. 예산을 "iter 고정"에서
  **"커리큘럼 도달 고정"** 으로 바꾸는 설계 변경이 유효한 해법(미검증).

### 6-I. terrain_level 보정 제안 (철회)
`terrain_level` 은 **사후(post-treatment) 매개변수**다 — 팔이 바로 그 변수에서 갈리므로
(G 7.961 vs N 7.495) 통제하면 인과경로가 정의상 삭제된다. 게다가 거의 순환: 학습 로그가 스스로
`terrain_level` 을 "crossed 0.8*corridor and survived" 라 적는다.

### 6-J. R.pet 1차 험지 스윕 (폐기)
27셀을 돌렸으나 두 결함으로 폐기: ①`--yaw` 를 ±1.5 랜덤으로 둬 제자리 선회 가능 ②이동거리를
아예 재지 않음. IsaacLab 의 gap·stepping 은 둘 다 **중앙에 평지 플랫폼**이 있고 로봇이 거기서
스폰하므로, 둘이 겹치면 "지형을 만나지 않은 결과"를 성능으로 오독한다.

---

## 7. 측정 함정 체크리스트 (새 측정 전 반드시)

1. **"확인"과 "추론"을 섞지 말 것.** 추론이면 "추정"이라고 먼저 밝힌다.
2. **도구 출력의 제약 조건을 확인.** `wc -l` 이 센 것이 데이터인가 에러인가.
   `pip index` 가 보여주는 것이 전체인가 현재 파이썬에 맞는 것만인가. 문서의 값이 요구사항인가
   남의 머신 상태인가.
3. **개입이 실제로 걸렸는지 재는 출력을 먼저 넣을 것** (배너 · `|p−psnap|` 같은 것).
   이 세션에서 게이트가 켜졌는데 적용 안 된 사례, 솔버가 둘인데 호출 안 되는 쪽을 고친 사례가 있다.
4. **`git add <디렉터리>` 는 ignored 파일을 경고 없이 건너뛴다.** 이 번들을 커밋할 때
   루트 `.gitignore:73 *.txt` 와 `:74 results/` 가 결과 로그 3개 + diffstat 1개를 조용히 뺐고,
   나는 "결과 원자료 포함"이라고 보고했다. `git add -f` 로 고쳤다.
   **번들을 커밋한 뒤에는 반드시 `디스크 파일 수 == git ls-files 수` 를 대조할 것:**
   ```bash
   comm -23 <(find <dir> -type f | sort) <(git ls-files <dir> | sort)
   ```
5. **집계 스크립트가 조용히 데이터를 버린다.** `rs_table_arms.py` 는 d-계열 전용으로 작성돼
   새 정책명마다 스킵했다(3번 겪음: 정규식 `(d\d+)` · `SEEDS` 고정집합 · `arm[A-Z]+` 가 숫자 거부).
   **새 팔 추가 시 롤아웃 수를 먼저 대조할 것.**
6. **주차/제자리 오염.** 속도 추종률만 보면 안 된다. **반드시 이동거리와 함께.**
   (stepping L8: 추종 94.3% · 이동 9 cm)
7. **`terrain_level` 은 판정에 쓰지 않는다.** 속도를 점수로 세고 순위가 뒤집히고 평균창에 따라
   부호가 바뀐다(이 라인에서 4번 깨졌다).
8. **추론 단위 = 학습 seed.** 롤아웃 단위 풀링은 귀무 하 위양성률 L8 16.9% · L9 57.4%.
9. **대응검정 금지** — 같은 학습seed 쌍 상관이 **음수**(L9 r=−0.60)라 검정력을 깎는다.
   비대응 MWU 정확검정을 쓸 것.
10. **seed 수를 정할 때 달성 가능한 최소 p 를 먼저 계산.** n=5 부호검정은 5/5 여도 양측 0.0625.
11. **중간 체크포인트로 판정하지 말 것 — 40k 이후만.** (09-09 문서 §A: 이 프로젝트에서 뒤집힌
    사례 4건. `cmdchg4s` gallop 20k 0.0% → 50k 37.9% 등)
12. **벽시계와 solve-only 를 섞어 비교하지 말 것** (8× 라 오보했다가 정정).
13. **종료 조건 없는 play 는 무한 재생된다.** `GO2_FH_MAX_STEPS` 빠뜨려 1 GB 덤프를 냈다.

---

## 8. 미측정으로 남은 것

**R.pet**
- 고속(vx 2~3) 걸음 분류 — 분류기 저속만 검증
- 접촉 스케줄(TAMOLS 결합 전제) — 센서 죽어 있음
- 난도 L2/L5/L8 은 **내가 정한 자체 스케일** — Go2 레벨과 직접 비교 불가.
  정의는 `rpet_my_edits.diff` 안에 그대로 있다(`f = L/9`):

  | 지형 | 파라미터 (f = L/9) | L2 | L5 | L8 |
  |---|---|---|---|---|
  | rough | `noise_range = (0, 0.02 + 0.08·f)` | ±3.8 cm | ±6.4 cm | ±9.1 cm |
  | gap | `gap_width = (0.1+0.3f, 0.15+0.35f)` | 17~23 cm | 27~34 cm | 37~46 cm |
  | stepping | `stone_width = (0.40−0.28f, 0.45−0.30f)` | 34~38 cm | 24~28 cm | 15~18 cm |
  | stepping | `stone_distance = (0.02+0.16f, 0.05+0.18f)` | 6~9 cm | 11~15 cm | 16~21 cm |

  **Go2 커리큘럼은 난도가 학습 중 자동 상승하는 `terrain_level` 이고 이쪽은 고정 투입**이라
  숫자가 같아도 의미가 다르다. 교차 인용 금지.
- 09-09 §G-4 거동 리스크: **발 형상**(우리 17-DOF 는 sphere 발 필수였는데 `Leg_gen` 은 URDF mesh) ·
  self-collision convex-hull 과대 · 게인 차이(회사 calf Kp 134/foot 59 vs 우리 12/20)
- 발목 토크 불일치: `rga_leg.py:35` **142.8** vs 우리 실기 Peak **100.8**(1.42배).
  09-09 §G 결정 = **우리 100.8 을 쓴다**(CFG 값이라 데이터셋 정합 안 깨짐)
- **관절 sign/offset 맵 미확정** — 우리 MJCF 는 뒷다리 thigh/calf·HR_foot 부호가 URDF/USD 와
  반대이고 range 도 다르다. TAMOLS 캐시를 어느 기구학으로 생성할지가 여기 걸린다
- 누락 파일 2개(송신자 요청 필요, 지금은 불요): `scripts/real2sim/export_deployable_leg17.py`
  (`spec.json:4` 가 가리킴 — fine-tune 후 재export 때 필요)

**Go2 / TAMOLS**
- 경사 징검돌(**비평면 희소지지**) — **발판 결론이 뒤집힐 수 있는 유일한 후보.**
  경사가 ①마찰추를 처음으로 구속 ②접촉을 진짜 비평면으로 만들어 GIAC 의 원래 조건을 성립시킴
  ③"최근접 돌"이 도달불가가 될 수 있게 함. 작업의 본체는 `_gap_surface_z` **스칼라 가정 25곳 해체**
- zero-shot 신규 지형 · 제약 보증/인증
- "발판 지각층 + 몸통만 TO" 분해로는 **실시간 불가 확정**(완벽한 발판을 줘도 33% 뿐, 필요 10×).
  정밀지형 20-iter solve = **4.4~4.7 초**. 평지 벤치 12.7 ms 는 **350배 낙관**(대표성 없음)

---

## 9. 관련 메모리(이 세션에서 작성)

`/home/jsh/.claude/projects/-home-jsh----jsh/memory/` 아래:
- `tamols-solve-cost-wall.md` — solve 4.5초 벽 · 분해 기각 · 측정 함정 4종
- `foothold-snap-confound.md` — 스냅이 제2변수였던 사례 · 결론 철회
- `rl-arm-comparison-methodology.md` — 학습seed=추론단위 · 대응검정 금지 · BASE_POLICY 미사용

다른 머신에서는 이 메모리가 없으니 **이 문서 §6·§7 이 그 역할을 한다.**

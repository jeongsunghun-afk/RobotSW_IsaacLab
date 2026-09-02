# PACE sim-to-sim — HindLeg 태스크 파일 (서버 배선 백업)

이 폴더는 GPU 서버(192.168.1.205)에서 PACE 액추에이터 식별 파이프라인을 우리
8-DOF biped `hind_leg`에 이식한 파일들의 로컬 백업이다. sim-to-sim 검증(주입한
GT 파라미터를 CMA-ES가 복원하는지) 결과는 `../PACE_통합_준비.md`의
"## sim-to-sim 검증 결과" 섹션 참조.

## 서버 파일 위치 (원본)
- `hindleg_pace_env_cfg.py`
  → `/mnt/ssd1/jsh/pace-sim2real/source/pace_sim2real/pace_sim2real/tasks/manager_based/pace/hindleg_pace_env_cfg.py`
  (설치된 pace_sim2real 패키지 안. RobotSW_IsaacLab/source 는 건드리지 않음.)
- `data_collection_hindleg.py`, `fit_run.py`, `report.py`
  → `/mnt/ssd1/jsh/pace_run/`  (실행 스크립트)
- `gt.pt` = 주입한 ground-truth 파라미터(검증 기준값). 실제 데이터로 바꿔 돌릴 때는 불필요.

## 태스크 등록 (서버에서 한 일)
`/mnt/ssd1/jsh/pace-sim2real/source/pace_sim2real/pace_sim2real/tasks/manager_based/__init__.py`
끝에 아래를 append 했다:

```python
gym.register(
    id="Isaac-Pace-HindLeg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pace.hindleg_pace_env_cfg:HindLegPaceEnvCfg"
    },
)
```

## 새 서버에 다시 깔 때
```bash
git clone https://github.com/leggedrobotics/pace-sim2real.git /mnt/ssd1/jsh/pace-sim2real
# isaac-5.1 conda env 안에서
pip install cmaes
pip install -e /mnt/ssd1/jsh/pace-sim2real/source/pace_sim2real
# 이 폴더의 hindleg_pace_env_cfg.py 를 pace 패키지 tasks/manager_based/pace/ 로 복사
# 위 gym.register 스니펫을 tasks/manager_based/__init__.py 에 append
# data_collection_hindleg.py / fit_run.py / report.py 를 /mnt/ssd1/jsh/pace_run/ 로 복사
```

## 핵심 이식 포인트 (anymal 예제 대비 바꾼 것)
1. 로봇 = `HIND_LEG_CFG` (isaaclab_assets.robots.rga), 액추에이터를 `PaceDCMotorCfg`
   8관절(`.*_hip/_thigh/_calf/_foot_joint`)로 교체. Kp/Kd 는 HIND_LEG_CFG 값 유지
   (65/53/12/20, 6/4.8/1.1/1).
2. `saturation_effort` 는 **스칼라**여야 함(DCMotor 가 `effort_limit/saturation_effort`
   를 raw 로 씀 → dict 주면 `Tensor / dict` 에러). 60.0 으로 설정(최대 effort 56 이상).
3. GT·탐색 bounds 를 우리 소관절 스케일로 축소: armature 1e-4~0.15, viscous 0~0.5,
   coulomb 0~0.15, bias ±0.06, delay 0~10. (anymal 은 armature 상한 1.0 으로 과대.)
4. joint_order 는 cfg 에서 비워두고(`[]`) 런타임에 articulation.joint_names 에서 유도
   → 정렬 실수 원천 차단. 실제 순서 = hip,hip,thigh,thigh,calf,calf,foot,foot (HL/HR).

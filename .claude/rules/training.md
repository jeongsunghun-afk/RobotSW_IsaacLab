# 학습·렌더 실행 규칙 (IsaacLab-6.0)

- conda 환경은 `isaac-6.0`. `./isaaclab.sh`는 conda를 스스로 활성화하지 않으므로 `conda activate isaac-6.0`을 먼저 한다.
- headless 실행은 `env -u DISPLAY`를 붙인다. SSH의 DISPLAY가 남아 있으면 GLXBadFBConfig로 죽는다.
- 학습에 `--video`를 붙이지 않는다. 영상은 체크포인트에서 `scripts/tools/report_video.py`로 렌더한다.
- 렌더는 학습이 점유하지 않은 GPU에서 한다. 같은 GPU면 흰 화면이 나온다.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`는 설정하지 않는다. PhysX 초기화가 멈춘다.
- `max_iterations`는 총 iter가 아니라 추가 iter 수다. 보고서에 그대로 적으면 틀린다.
- 학습·평가 진입점은 `scripts/reinforcement_learning/train.py`, `play.py`이며 `--rl_library rsl_rl`이 필수다. `scripts/reinforcement_learning/rsl_rl/train.py`는 남아 있지만 deprecated 경고를 내는 옛 경로다.
- 한 이터레이션에 원인 분리가 불가능할 만큼 여러 변경을 동시에 넣지 않는다.

```bash
source /home/user/miniconda3/etc/profile.d/conda.sh && conda activate isaac-6.0 && \
env -u DISPLAY ./isaaclab.sh -p scripts/reinforcement_learning/train.py \
  --rl_library rsl_rl --task <TASK> --num_envs <N> --headless
```

상세(소요 시간, exporter 버그 우회, 옛 체크포인트 호환): 글로벌 스킬 `isaaclab-train-video-flags`.

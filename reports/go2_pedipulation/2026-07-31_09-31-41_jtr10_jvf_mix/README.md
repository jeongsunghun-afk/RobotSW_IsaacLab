# 2026-07-31_09-31-41_jtr10_jvf_mix

gym task id: `Go2-Pedipulation-v0`

떨림 수정 sweep 의 한 arm. `2026-07-29_13-56-20_s2_circle/model_17598.pt` 에서 4000
iteration finetune (17598 → 21597), 2048 env, `trajectory_mode=circle`.

```
env.위
env.command.static_fraction=0.4
```

정지 목표를 학습 분포에 40% 섞음. 떨림 2.078 로 jtr10_jvf(1.991)와 구분 안 됨 → **분포 불일치 가설 기각.**

전체 비교와 판정 근거: [`../../_comparisons/tremor_regularization_sweep/`](../../_comparisons/tremor_regularization_sweep/)

산출물: `metrics/{hold,circle,pushhi}_dr_on.json`
원본 학습 로그: `logs/rsl_rl/go2_pedipulation/2026-07-31_09-31-41_jtr10_jvf_mix/`

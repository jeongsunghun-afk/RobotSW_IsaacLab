# 2026-07-30_10-26-19_jtr10_jvf

gym task id: `Go2-Pedipulation-v0`

떨림 수정 sweep 의 한 arm. `2026-07-29_13-56-20_s2_circle/model_17598.pt` 에서 4000
iteration finetune (17598 → 21597), 2048 env, `trajectory_mode=circle`.

```
env.w_joint_target_rate=-1.0
env.jvel_filter_alpha=0.25
```

둘 다. 떨림 1.991, push 는 base 이상. jtr40_jvf 로 대체됨.

전체 비교와 판정 근거: [`../_comparisons/tremor_regularization_sweep/`](../_comparisons/tremor_regularization_sweep/)

산출물: `metrics/{hold,circle,pushhi}_dr_on.json`
원본 학습 로그: `logs/rsl_rl/go2_pedipulation/2026-07-30_10-26-19_jtr10_jvf/`

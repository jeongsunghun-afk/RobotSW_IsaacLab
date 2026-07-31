# 2026-07-30_10-26-12_jtr10

gym task id: `Go2-Pedipulation-v0`

떨림 수정 sweep 의 한 arm. `2026-07-29_13-56-20_s2_circle/model_17598.pt` 에서 4000
iteration finetune (17598 → 21597), 2048 env, `trajectory_mode=circle`.

```
env.w_joint_target_rate=-1.0
```

페널티 단독. 지지 다리는 2.17→1.47 로 줄었으나 **조작 다리는 4.87→4.95 로 오히려 증가**. 100/150 N push 가 base 보다 낮다(95.1/72.9 vs 98.2/78.4). 기각.

전체 비교와 판정 근거: [`../_comparisons/tremor_regularization_sweep/`](../_comparisons/tremor_regularization_sweep/)

산출물: `metrics/{hold,circle,pushhi}_dr_on.json`
원본 학습 로그: `logs/rsl_rl/go2_pedipulation/2026-07-30_10-26-12_jtr10/`

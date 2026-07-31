# 2026-07-30_10-36-24_jvf_only

gym task id: `Go2-Pedipulation-v0`

떨림 수정 sweep 의 한 arm. `2026-07-29_13-56-20_s2_circle/model_17598.pt` 에서 4000
iteration finetune (17598 → 21597), 2048 env, `trajectory_mode=circle`.

```
env.jvel_filter_alpha=0.25
```

필터 단독. 떨림은 −5% 에 그치지만 **200 N push +22.0%p, 300 N +20.8%p 를 추종 비용 0 으로** 얻는다. 떨림 목적으로는 기각, 그러나 이 여유가 jtr40_jvf 를 가능하게 했다.

전체 비교와 판정 근거: [`../_comparisons/tremor_regularization_sweep/`](../_comparisons/tremor_regularization_sweep/)

산출물: `metrics/{hold,circle,pushhi}_dr_on.json`
원본 학습 로그: `logs/rsl_rl/go2_pedipulation/2026-07-30_10-36-24_jvf_only/`

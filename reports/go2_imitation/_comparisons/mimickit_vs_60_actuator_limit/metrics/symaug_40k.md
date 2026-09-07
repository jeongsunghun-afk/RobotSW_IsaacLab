# symmetry data-aug arm — iter 40k 램프 (JUDGMENT POINT)

`logs/symaug_20k_compare.py` 가 생성한다. 손으로 고치지 말 것.
램프 조건: `--no_pace --no_push --num_envs 64`, 시드 0·1 (학습 cfg 와 맞춤).

## 추종 · 낙상

```
                         arm |            0.5            1.0            1.5            2.0            2.5            3.0            3.5            4.0
  symaug @40k  (data-aug ON) |  0.31( 74%)F 0  0.64( 96%)F 0  0.88(100%)F 0  1.44( 83%)F 0  2.18( 97%)F 0  2.90( 98%)F 0  3.42( 98%)F 0  3.83( 97%)F 1
  parent @40k  (no symmetry) |  0.31( 74%)F 2  0.58( 84%)F 2  0.85( 80%)F 2  1.43( 91%)F 2  2.01( 94%)F 2  2.76( 95%)F 2  3.47( 95%)F 2  3.92( 95%)F 2
           symaug @20k (ref) |  0.22( 41%)F 0  0.59( 77%)F 0  0.99( 89%)F 0  1.69( 91%)F 0  2.44( 93%)F 0  3.12( 93%)F 0  3.68( 93%)F 0  4.08( 93%)F 0
          control @40k (ref) |  0.34( 89%)F 0  0.66( 98%)F 0  1.06( 97%)F 0  1.54(100%)F 0  2.12(100%)F 0  2.89(100%)F 0  3.58( 91%)F 0  0.90( 39%)F 2
```

괄호는 달성률(명령 절반을 넘은 env 비율), `F` 는 낙상 %. 시드 평균.

## ★ 좌우 비대칭 — 이 arm 이 겨냥한 것

ROM 좌우차 = `|L-R| / ((L+R)/2)` 의 env 중앙값. 램프는 `vy=0 · yaw=0` 을 명령하므로
남는 좌우 사용 차이는 전부 비대칭이다.

```
                         arm |   cmd |    thigh     calf   torque |  n_up
  symaug @40k  (data-aug ON) |   2.5 |     6.1%     8.1%     9.6% |    64
  symaug @40k  (data-aug ON) |   3.0 |     8.9%     8.1%     8.3% |    64
  symaug @40k  (data-aug ON) |   3.5 |     6.2%     6.9%     6.4% |    64
  symaug @40k  (data-aug ON) |   4.0 |     6.0%     5.4%     5.7% |    64
  parent @40k  (no symmetry) |   2.5 |    11.6%    22.2%     8.5% |    63
  parent @40k  (no symmetry) |   3.0 |    23.8%    19.1%    23.1% |    62
  parent @40k  (no symmetry) |   3.5 |    33.0%    18.2%    25.0% |    62
  parent @40k  (no symmetry) |   4.0 |    26.7%    16.0%    23.5% |    62
           symaug @20k (ref) |   2.5 |     4.2%     5.9%     5.4% |    64
           symaug @20k (ref) |   3.0 |     4.4%     4.9%     3.8% |    64
           symaug @20k (ref) |   3.5 |     3.9%     4.1%     3.6% |    64
           symaug @20k (ref) |   4.0 |     4.0%     4.7%     3.3% |    64
          control @40k (ref) |   2.5 |     5.5%     8.6%     8.6% |    64
          control @40k (ref) |   3.0 |     5.7%    11.6%    10.9% |    64
          control @40k (ref) |   3.5 |     7.2%    13.3%    13.0% |    64
          control @40k (ref) |   4.0 |    14.8%    16.3%    16.7% |    63
```


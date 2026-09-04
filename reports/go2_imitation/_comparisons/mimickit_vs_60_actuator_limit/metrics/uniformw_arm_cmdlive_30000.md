# 클립 균등 샘플링 arm — iter 30000 램프 · cmdlive (command resampled every 2-7 s) (★판정 구간 아님)

`logs/plot_uniformw_arm.py` 가 생성한다. 손으로 고치지 말 것.

```
                           arm |           0.5           1.0           1.5           2.0           2.5           3.0           3.5           4.0
  control: length-proportional |  0.301( 69%)  0.513( 52%)  0.895( 70%)  1.356( 67%)  1.943( 64%)  2.816( 62%)  3.488( 61%)  1.953( 48%)
   treatment: uniform per clip |  0.201( 36%)  0.540( 61%)  0.840( 61%)  1.307( 62%)  1.999( 59%)  2.644( 58%)  3.179( 58%)  3.704( 58%)
```

괄호는 달성률(명령의 절반을 넘은 env 비율).

```
                           arm |     0.5     1.0     1.5     2.0     2.5     3.0     3.5     4.0   <- 낙상 %
  control: length-proportional |     7.8     9.4    28.1    32.8    35.9    35.9    39.1    39.1
   treatment: uniform per clip |    10.9    18.8    31.2    37.5    40.6    42.2    42.2    42.2
```

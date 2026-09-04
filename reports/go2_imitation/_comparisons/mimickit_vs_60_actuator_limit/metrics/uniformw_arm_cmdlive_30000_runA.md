# 클립 균등 샘플링 arm — iter 30000 램프 · cmdlive (command resampled every 2-7 s) (★판정 구간 아님)

`logs/plot_uniformw_arm.py` 가 생성한다. 손으로 고치지 말 것.

```
                           arm |           0.5           1.0           1.5           2.0           2.5           3.0           3.5           4.0
  control: length-proportional |  0.316( 62%)  0.562( 70%)  0.930( 80%)  1.415( 78%)  2.031( 73%)  2.849( 69%)  3.488( 66%)  2.429( 52%)
   treatment: uniform per clip |  0.167( 41%)  0.505( 53%)  0.754( 52%)  1.285( 56%)  1.953( 55%)  2.622( 55%)  3.196( 55%)  2.116( 50%)
```

괄호는 달성률(명령의 절반을 넘은 env 비율).

```
                           arm |     0.5     1.0     1.5     2.0     2.5     3.0     3.5     4.0   <- 낙상 %
  control: length-proportional |     4.7     6.2    15.6    21.9    26.6    26.6    29.7    29.7
   treatment: uniform per clip |    25.0    32.8    39.1    43.8    45.3    45.3    45.3    48.4
```

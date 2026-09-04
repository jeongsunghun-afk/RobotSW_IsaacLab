# 클립 균등 샘플링 arm — iter 40000 램프 · cmdlive (command resampled every 2-7 s) (★판정 구간 아님)

`logs/plot_uniformw_arm.py` 가 생성한다. 손으로 고치지 말 것.

```
                           arm |           0.5           1.0           1.5           2.0           2.5           3.0           3.5           4.0
  control: length-proportional |  0.296( 64%)  0.611( 73%)  1.026( 77%)  1.543( 78%)  2.074( 72%)  2.836( 72%)  3.551( 66%)  0.290( 19%)
   treatment: uniform per clip |  0.230( 47%)  0.547( 56%)  0.799( 59%)  1.327( 62%)  1.868( 58%)  2.642( 58%)  3.286( 56%)  3.703( 56%)
```

괄호는 달성률(명령의 절반을 넘은 env 비율).

```
                           arm |     0.5     1.0     1.5     2.0     2.5     3.0     3.5     4.0   <- 낙상 %
  control: length-proportional |     6.2     9.4    18.8    21.9    26.6    26.6    26.6    32.8
   treatment: uniform per clip |    20.3    23.4    29.7    37.5    42.2    42.2    43.8    43.8
```

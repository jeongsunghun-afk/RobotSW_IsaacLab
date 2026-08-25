# AMP style 가중치 축 — 최종 요약 (40k 이후 점만)

`logs/style_weight_summary.py` 가 생성한다. 손으로 고치지 말고 스크립트를 다시 돌릴 것.


## STOCK

```
                     arm  pts | cmd 2.5 vx    % | cmd 4.0 vx    %  base_h  h std  flipHz  cap%
     lerp 0.5 (baseline)    4 |      1.598   97 |      0.049    0   0.253  0.040     8.7     2
      lerp 0.6 (scratch)    4 |      1.894   98 |      3.836   95   0.374  0.070    11.2    13
  lerp 0.8 -> 0.6 (curr)    3 |      2.290  100 |      3.756  100   0.384  0.051    11.7    12
                lerp 0.8    4 |      2.404   98 |      3.937   97   0.279  0.072    12.7     7
     lerp 1.0 (no style)    0 | (40k 이후 점 없음 — 중단됨)
```

## PACE

```
                     arm  pts | cmd 2.5 vx    % | cmd 4.0 vx    %  base_h  h std  flipHz  cap%
     lerp 0.5 (baseline)    3 |      1.418   86 |      0.101    0   0.254  0.041     8.0     1
                lerp 0.8    4 |      2.396   97 |      3.969   97   0.335  0.084    12.8     8
```

그림: `figures/style_weight_summary.png`

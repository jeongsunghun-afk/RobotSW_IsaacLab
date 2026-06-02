# 3-Leg Gait 순위 0 검증 결과 (v3 — height_scanner ground_z)

- 체크포인트: `/home/lgb/IsaacLab/logs/rsl_rl/go2_parkour/2026-06-01_14-06-08_use_default_usd/model_21900.pt`
- 실행 시각: 2026-06-02 13:19:21
- num_steps: 2000  num_envs: 64  cal_steps: 200
- force_thresh: 2.0 N  z_thresh: 0.03 m  v_thresh: 0.2 m/s
- ground_z 소스: height_scanner ray_hits_w (nearest finite ray)

## Scanner Coverage Diagnostic

```
height_scanner cfg: offset_x=+0.375m, size=[1.6, 1.0]m
scanner grid (body-relative x): [-0.425, 1.175] m
scanner grid (body-relative y): [-0.500, 0.500] m
(실제 bbox는 stdout 참조)
```

## Ray Valid Ratio (발별 scanner 커버리지)

| foot | ray_valid% (main) | mean_clearance_planted (m) |
|------|------------------|---------------------------|
| FL_foot | 100.0% | 0.3212 |
| FR_foot | 100.0% | 0.2411 |
| RL_foot | 100.0% | 0.1351 |
| RR_foot | 100.0% | 0.0236 |

**ray_valid% 해석**: 낮으면 scanner가 해당 발을 커버하지 못해 kin 신호 무효.

## Sole Offset (자동 추정)

| foot | sole_offset (m) | cal_n |
|------|----------------|-------|
| FL_foot | 0.3276 | 8626 |
| FR_foot | 0.2405 | 6309 |
| RL_foot | 0.1532 | 1404 |
| RR_foot | 0.0231 | 7975 |

## Overall per-foot metrics

```
=== OVERALL per-foot contact metrics ===
metric                         FL_foot       FR_foot       RL_foot       RR_foot
--------------------------------------------------------------------------------
sensor_ratio                    0.6743        0.5130        0.1056        0.6022
kin_ratio                       0.3726        0.4623        0.1237        0.6117
kin_only_ratio                  0.0901        0.2304        0.0830        0.0272
sensor_only_ratio               0.3918        0.2811        0.0649        0.0177
both_ratio                      0.2825        0.2319        0.0408        0.5845
neither_ratio                   0.2356        0.2566        0.8114        0.3706
```

## Per-terrain breakdown

```
--- Terrain: flat (class 0)  n=26000 ---
metric                         FL_foot       FR_foot       RL_foot       RR_foot
--------------------------------------------------------------------------------
sensor_ratio                    0.7936        0.5436        0.2094        0.6496
kin_ratio                       0.7018        0.8117        0.1050        0.6490
kin_only_ratio                  0.1785        0.4099        0.0542        0.0126
sensor_only_ratio               0.2703        0.1419        0.1586        0.0132
both_ratio                      0.5233        0.4017        0.0508        0.6364
neither_ratio                   0.0279        0.0465        0.7364        0.3378
```

```
--- Terrain: hurdle (class 1)  n=26000 ---
metric                         FL_foot       FR_foot       RL_foot       RR_foot
--------------------------------------------------------------------------------
sensor_ratio                    0.6549        0.5455        0.0740        0.5882
kin_ratio                       0.2221        0.3885        0.1280        0.6042
kin_only_ratio                  0.0580        0.1801        0.0834        0.0368
sensor_only_ratio               0.4908        0.3371        0.0294        0.0208
both_ratio                      0.1641        0.2084        0.0446        0.5674
neither_ratio                   0.2871        0.2744        0.8426        0.3750
```

```
--- Terrain: step (class 2)  n=26000 ---
metric                         FL_foot       FR_foot       RL_foot       RR_foot
--------------------------------------------------------------------------------
sensor_ratio                    0.6752        0.5093        0.0997        0.6077
kin_ratio                       0.3620        0.4278        0.1251        0.6203
kin_only_ratio                  0.0935        0.2157        0.0795        0.0282
sensor_only_ratio               0.4067        0.2972        0.0542        0.0156
both_ratio                      0.2685        0.2122        0.0456        0.5921
neither_ratio                   0.2313        0.2750        0.8207        0.3641
```

```
--- Terrain: gap (class 3)  n=26000 ---
metric                         FL_foot       FR_foot       RL_foot       RR_foot
--------------------------------------------------------------------------------
sensor_ratio                    0.6651        0.5162        0.0672        0.6092
kin_ratio                       0.3408        0.3987        0.1523        0.6312
kin_only_ratio                  0.0619        0.2050        0.1221        0.0356
sensor_only_ratio               0.3862        0.3226        0.0370        0.0137
both_ratio                      0.2789        0.1936        0.0302        0.5955
neither_ratio                   0.2730        0.2788        0.8107        0.3552
```

```
--- Terrain: stair (class 4)  n=24000 ---
metric                         FL_foot       FR_foot       RL_foot       RR_foot
--------------------------------------------------------------------------------
sensor_ratio                    0.5748        0.4451        0.0756        0.5526
kin_ratio                       0.2248        0.2702        0.1071        0.5490
kin_only_ratio                  0.0561        0.1341        0.0750        0.0223
sensor_only_ratio               0.4061        0.3090        0.0435        0.0260
both_ratio                      0.1687        0.1361        0.0321        0.5267
neither_ratio                   0.3691        0.4208        0.8494        0.4251
```

## Flags

edge-underreporting 근거 약함.

## 3-leg gait asymmetry

- FL_foot: sensor_ratio=0.6743  kin_ratio=0.3726  both_ratio=0.2825
- FR_foot: sensor_ratio=0.5130  kin_ratio=0.4623  both_ratio=0.2319
- RL_foot: sensor_ratio=0.1056  kin_ratio=0.1237  both_ratio=0.0408
- RR_foot: sensor_ratio=0.6022  kin_ratio=0.6117  both_ratio=0.5845

## Cross-verdict

**VERDICT**: RL_foot의 kin_ratio(0.1237)도 낮음 → 정책이 이 발을 실제로 공중에 유지. edge-underreporting 근거 없음. 원인: reward shaping / gait pairing 부재.

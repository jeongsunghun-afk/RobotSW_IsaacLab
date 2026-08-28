# 보행 attractor 조사 — 램프에 gallop 이 없는 이유
# 생성: gait_classify.py / gait_survey_multienv.py / gait_survey_analyze.py / ref_motion_gait.py

## 참조 모션 클립의 보행 (root 6열 오프셋 보정 후)
```
leg_run0                      1.47s  2.05Hz  FR 0.79 HL 0.77 HR 0.91  -> gallop
leg_run0_mirror               1.47s  2.13Hz  FR 0.21 HL 0.10 HR 0.98  -> gallop
leg_run1                      2.95s  2.58Hz  FR 0.77 HL 0.74 HR 0.89  -> gallop
leg_run1_mirror               2.95s  2.20Hz  FR 0.23 HL 0.08 HR 0.96  -> gallop
leg_trot0                     3.47s  2.11Hz  FR 0.49 HL 0.02 HR 0.50  -> pace
leg_trot0_mirror              3.47s  2.12Hz  FR 0.51 HL 0.02 HR 0.53  -> pace
leg_walk                      3.30s  0.88Hz  FR 0.52 HL 0.66 HR 0.09  -> trot
leg_walk1                     0.62s  1.68Hz  FR 0.40 HL 0.52 HR 0.22  -> other
leg_walk1_stmr                1.30s  0.96Hz  FR 0.38 HL 0.47 HR 0.27  -> other
leg_walk_stmr                 5.67s  0.96Hz  FR 0.13 HL 0.62 HR 0.92  -> other
leg_walk_turn                 3.47s  1.25Hz  FR 0.51 HL 0.70 HR 0.20  -> other
leg_walk_turn_mirror          3.47s  1.23Hz  FR 0.49 HL 0.67 HR 0.19  -> other
leg_walk_turn_mirror_stmr     7.45s  0.75Hz  FR 0.59 HL 0.60 HR 0.13  -> trot
leg_walk_turn_stmr            7.45s  0.94Hz  FR 0.41 HL 0.48 HR 0.13  -> trot
```

## 4096 env 서베이 — 참조 상태 출발 (env 기본)
```
표본 4096 → 사용 4096 (리셋 0 · 낙상 0 · 명령변동 0 제외)

[명령 격자별 gallop 비율]  행=vx [m/s], 열=yaw [rad/s]
칸 = gallop% / pace% (n).  최빈값만 쓰면 소수 보행이 안 보여 비율로 낸다.

vx \ yaw                 [-1.5,-0.5)           [-0.5,+0.5)           [+0.5,+1.5)
--------------------------------------------------------------------------------
[0.0,0.5)             0% / 0% (n=63)        1% / 0% (n=85)        8% / 0% (n=89)
[0.5,1.0)                        n=1                   n=0        7% / 0% (n=67)
[1.0,1.5)           21% / 68% (n=91)      26% / 64% (n=78)       8% / 71% (n=85)
[1.5,2.0)          15% / 85% (n=215)     28% / 72% (n=200)     27% / 73% (n=225)
[2.0,2.5)          26% / 74% (n=231)     27% / 73% (n=208)     27% / 73% (n=217)
[2.5,3.0)          24% / 76% (n=214)     30% / 70% (n=205)     21% / 79% (n=219)
[3.0,3.3)           18% / 82% (n=87)      28% / 72% (n=85)      26% / 74% (n=78)

[전체 분포] n=2743
  pace      1815   66.2%
  gallop     600   21.9%
  stand      208    7.6%
  other       90    3.3%
  trot        30    1.1%

★ gallop/bound n=600 — vx 0.12~3.20 (중앙 2.29) · |yaw| 0.00~1.50 (중앙 0.74)
```

## 4096 env 서베이 — 정지 출발 (--all_stand)
```
표본 4096 → 사용 4096 (리셋 0 · 낙상 0 · 명령변동 0 제외)

[명령 격자별 gallop 비율]  행=vx [m/s], 열=yaw [rad/s]
칸 = gallop% / pace% (n).  최빈값만 쓰면 소수 보행이 안 보여 비율로 낸다.

vx \ yaw                 [-1.5,-0.5)           [-0.5,+0.5)           [+0.5,+1.5)
--------------------------------------------------------------------------------
[0.0,0.5)             2% / 0% (n=91)       0% / 0% (n=126)       10% / 0% (n=90)
[0.5,1.0)                        n=2                   n=0       19% / 0% (n=63)
[1.0,1.5)           0% / 88% (n=110)       0% / 78% (n=51)      0% / 78% (n=118)
[1.5,2.0)          0% / 100% (n=224)     0% / 100% (n=206)     0% / 100% (n=226)
[2.0,2.5)          0% / 100% (n=177)     0% / 100% (n=212)     0% / 100% (n=200)
[2.5,3.0)          0% / 100% (n=216)     0% / 100% (n=200)     0% / 100% (n=219)
[3.0,3.3)           0% / 100% (n=81)       1% / 99% (n=92)      0% / 100% (n=71)

[전체 분포] n=2775
  pace      2350   84.7%
  stand      273    9.8%
  other       91    3.3%
  trot        35    1.3%
  gallop      23    0.8%
  bound        3    0.1%

★ gallop/bound n=26 — vx 0.17~3.15 (중앙 0.62) · |yaw| 0.23~1.50 (중앙 1.40)
```

## 참조 클립: 평균 전진속도 vs 보행 (속도-보행 사다리)

```
clip                              속도      주파수  보행
--------------------------------------------------------
leg_walk_stmr                  0.11m/s   0.96Hz  other
leg_walk                       0.19m/s   0.88Hz  trot
leg_walk_turn_mirror_stmr      0.32m/s   0.75Hz  trot
leg_walk_turn_stmr             0.32m/s   0.94Hz  trot
leg_walk1_stmr                 0.45m/s   0.96Hz  other
leg_walk_turn                  0.68m/s   1.25Hz  other
leg_walk_turn_mirror           0.68m/s   1.23Hz  other
leg_walk1                      0.94m/s   1.68Hz  other
leg_trot0                      1.97m/s   2.11Hz  pace
leg_trot0_mirror               1.97m/s   2.12Hz  pace
leg_run0                       2.69m/s   2.05Hz  gallop
leg_run0_mirror                2.69m/s   2.13Hz  gallop
leg_run1                       3.03m/s   2.58Hz  gallop
leg_run1_mirror                3.03m/s   2.20Hz  gallop
```

참조상 gallop 은 2.7 m/s 이상의 걸음인데, 정지 출발 정책은 4.0 에서도 pace 다.

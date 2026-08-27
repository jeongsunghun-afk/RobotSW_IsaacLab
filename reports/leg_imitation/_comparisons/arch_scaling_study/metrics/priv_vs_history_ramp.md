# priv latent(학습 경로) vs history latent(배포 경로) 램프 A/B

체크포인트: logs/rsl_rl/leg_imitation_tracking_rma/2026-08-24_17-27-53_torque1e5_stand_dz_vmax32_ds14_wcmd/model_49999.pt
조건: --run_params(학습 env.yaml 복원) --force_stand --no_video, 4 repeat x 2 경로
원자료: _workspace/leg/ab_privhist/{hist,priv}_{1..4}/ramp_data.npz
생성: _workspace/leg/ramp_onset_duty.py

duty = hold 구간 뒤 60%에서 |jvel| RMS > 0.4 인 스텝 비율 (1.0=연속 보행, 0=정지)
ach  = median(vx) / cmd

## cmd 0.5
history                      up      0.54     37%    4
                             down    1.00     71%    4
priv                         up      0.21     17%    4
                             down    0.99     67%    4

## cmd 1.0
history                      up      1.00     75%    4
                             down    1.00     69%    4
priv                         up      1.00     71%    4
                             down    1.00     70%    4

## cmd 1.5
history                      up      1.00     69%    4
                             down    1.00     79%    4
priv                         up      1.00     68%    4
                             down    1.00     78%    4

## cmd 2.0
history                      up      1.00     96%    4
                             down    1.00     95%    4
priv                         up      1.00     94%    4
                             down    1.00     97%    4

## cmd 2.5
history                      up      1.00     91%    4
                             down    1.00     91%    4
priv                         up      1.00     90%    4
                             down    1.00     91%    4

## cmd 3.0
history                      up      1.00     87%    4
                             down    1.00     87%    4
priv                         up      1.00     86%    4
                             down    1.00     86%    4

## cmd 3.2
run                          dir     duty     ach    n
------------------------------------------------------
history                      (매칭된 npz 없음: _workspace/leg/ab_privhist/hist_*/ramp_data.npz)
priv                         (매칭된 npz 없음: _workspace/leg/ab_privhist/priv_*/ramp_data.npz)


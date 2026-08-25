# 2026-07-31_09-56-10_afilt_jvf — **hang 으로 폐기**

gym task id: `Go2-Pedipulation-v0`

`action_filter_alpha=0.35` arm 의 첫 시도. **iteration 19617 에서 3시간 동안 로그·체크포인트
없이 CPU 만 태우다 죽였다.** `videos/train/rl-video-step-1000.mp4` 가 남아 있어 학습 중 영상
녹화 경로가 원인이다 — 이 저장소에서 두 번째다(S1 본 학습도 같은 이유로 10h20m hang).

**교훈: 학습은 `--video` 없이 돌리고, 영상은 체크포인트에서 따로 렌더한다.**
(`reports/README.md` 의 "렌더러 사용법" 과 같은 결론)

재시도: `2026-07-31_14-33-07_afilt_jvf_r2` — `--video` 없이 같은 시작점에서 4000 iteration.

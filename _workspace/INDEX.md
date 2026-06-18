# _workspace 보고서 인덱스

**최종 갱신**: 2026-05-20  
**용도**: IsaacLab Parkour/Mimic/Locomotion 프로젝트의 분석·디버그·RCA 보고서 30여 개에 대한 빠른 탐색 가이드  
**갱신 방법**: 수동 (새 보고서 추가 시 해당 섹션에 수동 추가 권장)

---

## 📌 빠른 네비게이션

### 최신 통합 분석 (우선 확인)
- **Parkour A vs B 차이 종합**: [`parkour_diff_summary.md`](#parkour-a-vs-b-환경-비교-통합) — 2026-05-19, 전체 차이점 통합 보고서
- **Parkour 보행 품질 비교**: [`parkour_flat_gait_A_vs_B_analysis.md`](#parkour-a-vs-b-환경-비교-통합) — 2026-05-19, 평지 trot 품질 차이 원인 분석
- **Parkour Gait RCA**: [`parkour_gait_rca.md`](#1-parkour-환경-rca--디버그) — 2026-05-12, 3대 보행 결함(drag/yaw/twitch) root cause

### RCA 최종 결론 (실무 참조용)
- **좌측 뒷다리 비대칭 + 계단 정지**: [`parkour_left_hind_stair_freeze_RCA.md`](#1-parkour-환경-rca--디버그) — 2026-05-13 v2 (deploy 제약 반영)
- **Step/Stair/Gap 학습 실패**: [`parkour_step_stair_gap_RCA_summary.md`](#1-parkour-환경-rca--디버그) — 2026-05-12 통합본

---

## 1. Parkour 환경 RCA / 디버그

### 보행 결함 분석 (Gait Issues)
- [`parkour_gait_rca.md`](parkour_gait_rca.md) — `2026-05-12` 3대 증상(뒷다리 drag, yaw 진동, reset twitch) root cause. Tier A+B 증거 기반. Fix-1~4 제안
- [`parkour_left_hind_stair_freeze_RCA.md`](parkour_left_hind_stair_freeze_RCA.md) — `2026-05-13` v2 (deploy 제약 반영). 좌측 뒷다리 비대칭 + 계단 정지. contact-obs 경로 닫힘, Genesis divergence 조사 필요
- [`parkour_flat_trot_action_plan.md`](parkour_flat_trot_action_plan.md) — `2026-05-13` 평지 trot 학습 변경 권장서. feet_air_time reward 추가 + dof_error_l2 강화 + yaw wrap 복구 3종 batch

### Step/Stair/Gap 학습 실패
- [`parkour_step_stair_gap_RCA_summary.md`](parkour_step_stair_gap_RCA_summary.md) — `2026-05-12` 통합 보고서. H1: actuator_mode=2 약화가 1순위, terrain split 설명
- [`parkour_step_stair_gap_log_evidence.md`](parkour_step_stair_gap_log_evidence.md) — `2026-05-12` Log 증거(Tier A). run 5개 메트릭 비교
- [`parkour_step_stair_gap_rca_codeside.md`](parkour_step_stair_gap_rca_codeside.md) — `2026-05-12` Code 증거(Tier B). 8개 변경사항 영향 매트릭스

### Fix 적용 검증
- [`parkour_fix_application_audit.md`](parkour_fix_application_audit.md) — `2026-05-13` parkour_gait_rca.md 제안 Fix-1~4 중 어느 것이 HEAD에 적용됐는지 audit. Fix-3(contact obs) 미적용 확인
- [`parkour_indexing_audit.md`](parkour_indexing_audit.md) — `2026-05-13` Genesis vs Isaac 관절/발 인덱싱 순서 검증. FL↔FR, RL↔RR 스왑 확인

### 특정 Reward/Feature 검증
- [`parkour_feet_air_time_validation.md`](parkour_feet_air_time_validation.md) — `2026-05-13` feet_air_time reward 제안 검증. anymal_c/R_Skeleton 표준 패턴 bit-identical, GO 판정
- [`parkour_tracking_yaw_genesis_alignment.md`](parkour_tracking_yaw_genesis_alignment.md) — `2026-05-12` Genesis tracking_yaw 원전 정렬 분석. moving_mask 게이트 제거 commit 영향

### Goal System
- [`parkour_goal_system_report.md`](parkour_goal_system_report.md) — `2026-04-30` Goal waypoint 시스템 비교(IsaacLab/Genesis/RobotSW_Parkour). PARKOUR_GOALS_REGISTRY 패턴 설명
- [`parkour_gap_analysis.md`](parkour_gap_analysis.md) — `2026-04-30` 3개 reference 대비 gap 분석. rsl_rl 내 PPOParkour 자산 발견

### 기타 진단
- [`parkour_flat_run_diagnostic.md`](parkour_flat_run_diagnostic.md) — `2026-05-14` 평지 전용 학습 1865 iter 진단. tracking_goal_vel 성공, drag 128× 악화
- [`parkour_smoke500_analysis.md`](parkour_smoke500_analysis.md) — `2026-05-14` 500-iter smoke test 합격 판정. tracking_goal_vel 2.5배 상승

---

## 2. Parkour A vs B 환경 비교 (통합)

### 통합 보고서
- [`parkour_diff_summary.md`](parkour_diff_summary.md) — `2026-05-19` A(학습 안 됨) vs B(학습 잘 됨) 전체 차이 통합. H1: 음수 reward 누적 + base_contact 조기 종료 = suicide optimum 구조
- [`parkour_flat_gait_A_vs_B_analysis.md`](parkour_flat_gait_A_vs_B_analysis.md) — `2026-05-19` 평지 보행 품질 차이. Manager vs Direct 프레임워크 차이 아님, reward config divergence

### 세부 비교 (parkour_diff_summary.md 하위 보고서)
- [`parkour_diff_reward.md`](parkour_diff_reward.md) — `2026-05-19` worker-1. Reward 구조 + 종료 조건 차이
- [`parkour_diff_terrain_curriculum.md`](parkour_diff_terrain_curriculum.md) — `2026-05-19` worker-2. Terrain 생성 + curriculum 차이
- [`parkour_diff_goal.md`](parkour_diff_goal.md) — `2026-05-19` worker-3. Goal 시스템 차이
- [`parkour_diff_episode_dr.md`](parkour_diff_episode_dr.md) — `2026-05-19` worker-4. Episode/DR 차이
- [`parkour_diff_control_hyperparam.md`](parkour_diff_control_hyperparam.md) — `2026-05-19` worker-5. Control/하이퍼파라미터 차이

### 특화 비교
- [`parkour_algorithm_hyperparam_A_vs_B.md`](parkour_algorithm_hyperparam_A_vs_B.md) — `2026-05-19` 알고리즘/하이퍼파라미터 차이. 핵심: 알고리즘 동일, estimator 사용 여부만 차이
- [`parkour_terrain_generation_comparison.md`](parkour_terrain_generation_comparison.md) — `2026-05-19` Terrain 생성 방식 비교. A=box primitive 직접 조립, B=height-field rasterize

---

## 3. Parkour DR/Priv/Estimator 연구

- [`parkour_dr_research.md`](parkour_dr_research.md) — `2026-05-14` Domain Randomization 카탈로그. IsaacLab DR 함수 + Go2/Anymal-C 표준 DR 세트 비교
- [`parkour_priv_research.md`](parkour_priv_research.md) — `2026-05-14` Privileged observation 확장 연구. RMA 패턴, 현재 14D → mass/COM/actuator randomization 추가 제안
- [`parkour_estimator_design_contract.md`](parkour_estimator_design_contract.md) — `2026-05-19` A 환경에 B-스타일 Estimator 도입 계약. obs dict 키 설계 `priv_explicit`/`priv_latent` 분리

---

## 4. Mimic / 모방학습 관련

### IsaacLab vs MimicKit 비교
- [`mimic_comparison.md`](mimic_comparison.md) — `2026-04-02` AMP 구현 비교. discriminator 구조/loss/GP/replay buffer 차이 분석
- [`Isaac_mimickit_comparision.md`](Isaac_mimickit_comparision.md) — `2026-04-XX` IsaacLab vs MimicKit 코드베이스 비교. 프레임워크/observation/종료조건 차이
- [`Isaac_mimickit_comparision2.md`](Isaac_mimickit_comparision2.md) — `2026-04-XX` Go2 AMP 학습 실패 원인 정밀 분석. Replay buffer 미적용 + 종료조건 관대함 + LS-GAN vs BCE

> **통합/최신**: `Isaac_mimickit_comparision2.md`가 가장 상세. `mimic_comparison.md`는 알고리즘 중심, `Isaac_mimickit_comparision.md`는 환경 중심.

### MimicKit 참고 자료
- [`mimickit_analysis.md`](mimickit_analysis.md) — `2026-04-XX` MimicKit 프레임워크 분석. 지원 알고리즘(DeepMimic/AMP/AWR/ASE/LCP/ADD), 디렉토리 구조

---

## 5. 기타 로봇/알고리즘 참고

- [`smooth_humanoid_locomotion_analysis.md`](smooth_humanoid_locomotion_analysis.md) — `2026-04-XX` LCP(Lipschitz-Constrained Policies) + PPO-RMA 분석. Gradient Penalty로 부드러운 보행, Isaac Gym 기반
- [`02_advisor_final_proposal.md`](02_advisor_final_proposal.md) — `2026-04-XX` BCAMP(Behavior-Controllable AMP) 연구 적용 제안서. Command-conditioned discriminator, RSL-RL 호환성 상

---

## 6. Plans / 제안서

**하위 폴더**: [`plans/`](plans/)

- [`plans/parkour_phase0_patch_draft.md`](plans/parkour_phase0_patch_draft.md) — `2026-05-07` 신규 6종 지형 추가 사전 골격 패치 초안
- [`plans/parkour_new_terrains_plan.md`](plans/parkour_new_terrains_plan.md) — `2026-05-07` 신규 지형(crawl, slope, zigzag_hurdles, rough_blocks 등) 설계 계획
- [`plans/parkour_crawl_plan.md`](plans/parkour_crawl_plan.md) — `2026-05-07` Crawl 지형 상세 설계
- [`plans/parkour_terrain_critic_review.md`](plans/parkour_terrain_critic_review.md) — `2026-05-07` Terrain 확장 비평 검토
- [`plans/parkour_terrain_extension_summary.md`](plans/parkour_terrain_extension_summary.md) — `2026-05-07` Terrain 확장 요약

---

## 7. 세부 비교 데이터 (하위 폴더)

### parkour_experiment_review/ (2026-05-14)
실험 영상 분석 + 파라미터 변경 추적

- `REPORT.md` — 종합 보고서
- `01_params_changes.md` — 파라미터 변경 이력
- `02_terrain_metrics.md` — Terrain별 메트릭
- `03_video_motion.md` — 영상 모션 분석
- `frames/` — 영상 프레임 캡처

### parkour_reward_compare/ (2026-05-14)
Reward 정의 라인별 비교

- `COMPARISON.md` — 종합 비교표
- `external_rewards.md` — 외부 reference reward
- `isaaclab_rewards.md` — IsaacLab parkour reward

### parkour_sim_cfg_compare/ (2026-05-14)
Simulation config 비교

- `COMPARISON.md` — 종합 비교표
- `external_sim_cfg.md` — 외부 reference sim config
- `isaaclab_sim_cfg.md` — IsaacLab sim config

---

## 8. 디버그 보고서 (일반)

- [`debug_report.md`](debug_report.md) — `2026-04-09` go2_imitation AMP discriminator policy loss 0 고착. `.clone()` 누락으로 메모리 aliasing → mode collapse
- [`report_2026-04-06.md`](report_2026-04-06.md) — `2026-04-06` Go2 WTW LCP Gradient Penalty 버그 3종 수정. priv_latent 독립변수 취급 오류 등

---

## 9. 보고서 갱신 이력

| 날짜 | 주요 추가 보고서 | 비고 |
|------|-----------------|------|
| 2026-05-19 | parkour_diff_summary.md + 5개 하위 보고서 | A vs B 통합 분석 완료 |
| 2026-05-14 | parkour_dr_research.md, parkour_priv_research.md | DR/Priv 연구 시리즈 |
| 2026-05-13 | parkour_left_hind_stair_freeze_RCA.md v2 | Deploy 제약 반영 |
| 2026-05-12 | parkour_gait_rca.md, parkour_step_stair_gap 3종 | 주요 RCA 완료 |
| 2026-04-30 | parkour_goal_system_report.md, parkour_gap_analysis.md | Goal 시스템 분석 |
| 2026-04-09 | debug_report.md | go2_imitation clone() 버그 |
| 2026-04-06 | report_2026-04-06.md | WTW LCP 버그 수정 |

---

## 10. 중복/Superseded 후보 (임시 참고용)

다음 보고서들은 여러 버전이 존재하거나 통합본으로 대체 가능:

- **Isaac_mimickit_comparision.md** ← `Isaac_mimickit_comparision2.md`가 더 상세 (2는 학습 실패 원인 중심)
- **parkour_step_stair_gap_log_evidence.md** + **parkour_step_stair_gap_rca_codeside.md** ← `parkour_step_stair_gap_RCA_summary.md`가 통합본

> **주의**: 위 파일들을 삭제/이동하지 마세요. 각 보고서는 증거 레벨(Tier A/B)이 다르므로 cross-reference 가치가 있습니다.

---

## 📖 보고서 읽는 법

### Tier 표기 의미 (RCA 보고서)
- **Tier A**: Log/metric 정량 증거 (가장 강함)
- **Tier B**: Code-surface diff, 구조 비교
- **Tier C**: 이론적 추론 (단독으로 1순위 금지)

### 금지 영역 (Parkour 분석)
프로젝트 메모리 `project_parkour_analysis_constraints.md` 참조:
1. Contact sensor를 policy obs에 추가 금지 (sim-to-real 제약)
2. Genesis 없는 새 reward 도입 금지
3. Reward/hyperparameter scale 튜닝 단독 액션 금지
4. Action latency 무관
5. `_actuator_mode=2` blame 금지
6. 토크 envelope 정량 추론 Tier-2 금지
7. Height_scan 의심 금지

---

**마지막 갱신**: 2026-05-20 (INDEX.md 생성)  
**다음 갱신 시점**: 새 RCA/비교 보고서 추가 시 해당 섹션에 수동 추가

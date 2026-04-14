## 디버그 분석 결과
- 환경: go2_imitation
- 학습 유형: IL (AMP)
- 주요 증상: policy가 Discriminator를 속이지 못해 discriminator의 policy loss가 시간이 지나도 상승하지 않는(0으로 고착되는) 문제
- 분석 일시: 2026-04-09

### 진단 내역
1. **[발견된 문제]** `extras["amp_obs"]` 반환 시 `.clone()` 누락으로 인한 메모리 공유(Aliasing) 및 Policy 배치 모드 붕괴 (go2_imitation_env.py:155)
   - 원인: `_get_observations`에서 `self.extras["amp_obs"] = self.amp_observation_buffer.view(-1, self.amp_observation_size)`를 통해 반환하는데, 이는 버퍼의 View(참조)입니다. 러너(`OnPolicyRunnerAMPBase`) 루프에서 이를 `detach()`하여 `amp_obs_buffer` 리스트에 24번 누적하지만, `detach()`는 메모리를 복사하지 않습니다. 
   매 스텝마다 환경 버퍼가 `self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]`로 In-place 업데이트 되므로, 러너에 누적된 24개의 관측치들이 모두 마지막 24번째 스텝의 상태(또는 RSI 리셋 상태)로 덮어씌워지게 됩니다. 
   결과적으로 Discriminator가 학습하는 Policy 배치(`24 * N`)는 실제로는 `N`개의 샘플이 24번 중복된 심각한 Mode Collapse 상태가 되며, 다채로운 Expert 데이터와 너무나 쉽게 구분(Perfect separation)되어 Policy Loss가 즉시 0으로 고착됩니다.
   - 심각도: critical
   - 권장 조치: `go2_imitation_env.py`의 `_get_observations` 마지막 부분에서 `self.amp_observation_buffer.view(...).clone()`을 호출하여 독립된 메모리 텐서로 반환하도록 수정하세요. (필요 시 `go2_amp_env.py`에도 동일하게 적용)

2. **[발견된 문제]** AMP Observation 차원 및 관절/발(Foot) 순서 검증 (정상)
   - 원인: 
     - shape 계산: `43(per step) × 10(history) = 430`으로 config와 완벽히 일치.
     - `dof_pos`, `dof_vel`: `_motion_dof_indices`를 사용해 Expert 모션 데이터를 IsaacLab의 알파벳 관절 순서와 일치시킴.
     - `foot_pos`: Policy는 `KEY_BODY_NAMES = ["FL_foot", "FR_foot", ...]`를 사용하고, Expert(`motion_lib.py`의 `_go2_fk_foot_pos`)도 동일한 `["FL", "FR", "RL", "RR"]` 순서로 역기구학을 연산하여 불일치 없음.
     - `rsi`: 동일한 `motion_ids` 및 `times`를 사용해 로봇 상태와 AMP 버퍼가 정확히 일치함.
   - 심각도: info
   - 권장 조치: AMP 구조 자체는 MimicKit과 수학적으로 완벽히 동일하므로 추가적인 알고리즘 수정은 불필요합니다. `.clone()` 버그만 수정하면 정상 학습될 것입니다.

### 체크리스트 결과
| 항목 | 결과 | 근거 |
|------|------|------|
| `num_amp_observations` 및 버퍼 차원 확인 | PASS | 10프레임 × 43차원 = 430차원으로 정상 계산됨 (`go2_imitation_env_cfg.py`) |
| Discriminator 구조 (`input_dim`) 확인 | PASS | `amp_observation_space=430` 이 PPOAMPBase 초기화 시 정상 전달됨 (`rsl_rl_ppo_cfg.py`) |
| AMP Reward 계산 및 clamp 로직 확인 | PASS | `amp_discriminator.py`의 `clamp(1 - 0.25 * (D - 1)^2, min=0)` 정상 구현됨 |
| Expert / Policy 간 Joint 위치 매핑 확인 | PASS | `_motion_dof_indices`를 통해 Policy(`robot.data.joint_pos`)와 Expert(`dof_pos`) 순서가 정확히 매핑됨 |
| RSI와 AMP 버퍼 간 `motion_ids` 일치 여부 | PASS | `_reset_strategy_rsi`에서 `motion_ids`와 `times`가 완벽히 동일하게 `collect_reference_motions`에 전달됨 |
| Runner에서 AMP Obs 독립적 수집 여부 | FAIL | `extras["amp_obs"]`가 `.clone()` 없이 참조로 반환되어 Runner 누적 시 메모리 덮어쓰기(Aliasing) 발생 |

### 피드백/교훈
- [2026-04-09] PyTorch 환경에서 `.view()`나 인덱싱으로 반환된 텐서를 Runner 루프의 리스트에 반복 누적할 때는 반드시 `.clone()`을 수행해야 함. `.detach()`는 Autograd 그래프만 분리할 뿐 메모리 스토리지를 공유하므로, 환경 내 In-place 업데이트가 발생하면 러너의 과거 데이터가 모두 손상(마지막 상태로 동기화)되어 심각한 Mode Collapse를 유발함.
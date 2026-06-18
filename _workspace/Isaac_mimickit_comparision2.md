# IsaacLab vs MimicKit: AMP Go2 학습 실패 원인 정밀 분석 보고서

질문해주신 내용을 바탕으로 `go2_amp`에서 학습이 실패하고 `MimicKit`에서 성공하는 원인을 파악하기 위해, MimicKit의 설정 파일 및 코드 레벨의 디테일한 차이점을 분석했습니다.

가장 치명적인 차이점은 **Replay Buffer의 실제 적용 여부**와 **종료 조건(Termination)의 관대함**, 그리고 **Loss 함수의 형태**에 있습니다.

---

## 1. MimicKit AMP Go2 관련 설정 파일 분석 (`args`, `yaml`)

MimicKit의 Go2 학습 환경을 정의하는 핵심 파일들은 다음과 같습니다.

*   **`args/amp_go2_args.txt`**
    *   `--num_envs 4096`: 4096개의 병렬 환경을 사용.
    *   `--agent_config data/agents/amp_go2_agent.yaml`, `--env_config data/envs/amp_go2_env.yaml`를 로드함.
*   **`data/agents/amp_go2_agent.yaml`**
    *   **네트워크 구조**: Actor, Critic, Discriminator 모두 `fc_2layers_1024units` (은닉층 2개, 1024개 노드).
    *   **옵티마이저**: 모두 **SGD** 사용. (IsaacLab의 Adam과 다름)
        *   Actor LR: `2e-4`, Critic LR: `1e-4`, Disc LR: `2.5e-4`
        *   Discriminator에 `weight_decay: 0.0001` 적용.
    *   **AMP 하이퍼파라미터**:
        *   `disc_buffer_size: 200000` (Replay Buffer 크기)
        *   `disc_replay_samples: 1000` (매 스텝 버퍼에 추가할 샘플 수)
        *   `disc_grad_penalty: 5.0`, `disc_reward_scale: 2.0`
    *   **보상 가중치**: `task_reward_weight: 0.0`, `disc_reward_weight: 1.0`. (오직 모방 보상만으로 학습)
*   **`data/envs/amp_go2_env.yaml`**
    *   **Observation**: `global_obs: True`, `root_height_obs: True`, **`num_disc_obs_steps: 10`** (10 프레임의 과거 상태를 누적).
    *   **종료 조건**: **`pose_termination: False`** (매우 중요).
    *   **접촉 및 주요 부위**: `key_bodies` & `contact_bodies` 모두 `["FR_foot", "FL_foot", "RR_foot", "RL_foot"]` 적용.

---

## 2. MimicKit Observation의 정확한 구성

MimicKit의 Discriminator Observation은 `mimickit/envs/deepmimic_env.py`의 `compute_tar_obs`, `compute_disc_vel_obs`, `compute_deepmimic_obs` 함수에서 조립됩니다. 단일 프레임 상태를 1D 벡터로 편 뒤, `num_disc_obs_steps: 10`에 의해 10개의 프레임이 합쳐집니다(Concat).

한 프레임당 Observation에 들어가는 값들은 다음과 같습니다:
1.  **`root_pos_obs` (3D)**: 기준(원점) 대비 Root의 `(x, y, z)` 위치 (`root_height_obs: True`로 높이도 포함).
2.  **`root_rot_obs` (6D)**: Root의 Quaternion 회전을 `quat_to_tan_norm` (Tangent-Normal) 6D 연속 표현으로 변환한 값.
3.  **`joint_rot_obs` (12 $\times$ 6D = 72D)**: 12개 관절의 회전(Quaternion)을 각각 6D로 변환한 값.
4.  **`key_pos` (4 $\times$ 3D = 12D)**: 발 4개의 상대적 위치.
5.  **`root_vel_obs` (3D) / `root_ang_vel_obs` (3D)**: 베이스의 선속도와 각속도 (`global_obs: True` 설정에 따라 로컬 좌표계 변환 없이 그대로 들어감).
6.  **`dof_vel` (12D)**: 12개 관절의 속도.

> **IsaacLab과의 차이**: IsaacLab은 43차원(12+12+1+3+3+12) 벡터인 반면, MimicKit은 회전 각도를 6D로 변환하는 등 차원 수가 훨씬 크고, 이를 10프레임 통째로 넣어 판단의 해상도를 높입니다.

---

## 3. Pose Termination (자세 이탈에 의한 종료) 여부

**가장 큰 학습 양상의 차이를 만드는 부분입니다.**

*   `data/envs/amp_go2_env.yaml` 코드를 확인하면 **`pose_termination: False`** 로 명시되어 있습니다.
*   즉, MimicKit은 기준 모션(Reference)과 로봇의 자세(Key body 위치 등)가 아무리 크게 벌어져도 **강제로 에피소드를 종료시키지 않습니다.** 로봇은 이상한 자세로 허우적거리더라도 페널티(낮은 보상)를 받으며 에피소드 길이(10초) 끝까지 버티면서 어떻게든 스스로 자세를 교정하는 법을 배우는 '탐험(Exploration)' 과정을 거치게 됩니다.
*   반면 IsaacLab은 `pose_termination`이 활성화되어 있어 자세가 무너지면 바로 죽어버리므로(Reset), 에이전트가 실패 상태에서 회복하는 데이터를 수집하지 못해 오히려 학습이 조기 수렴(Local Minima)에 빠졌을 가능성이 높습니다.

---

## 4. 학습 알고리즘 (Loss 함수) 수식 비교

### MimicKit (`amp_agent.py`) - Standard GAN (BCE)
MimicKit은 `BCEWithLogitsLoss`를 사용하여 Expert를 1, Policy를 0으로 이진 분류합니다.

$$ L_{disc} = \frac{1}{2} \Big[ \text{BCE}(D(x_{policy}), 0) + \text{BCE}(D(x_{expert}), 1) \Big] $$
$$ + \lambda_{gp} \Big( \| \nabla_{x_{expert}} D \|^2 + \| \nabla_{x_{policy}} D \|^2 \Big) + \lambda_{logit} \sum w_{disc}^2 $$

*   **Logit 정규화 (`disc_logit_reg`)**: 신경망의 가중치($w_{disc}$) 자체에 L2 정규화를 가합니다 (`torch.sum(torch.square(logit_weights))`).
*   코드 위치: `mimickit/learning/amp_agent.py` 의 `_compute_disc_loss()`

### IsaacLab (`ppo_amp.py`) - Least Squares GAN (MSE)
IsaacLab은 `MSELoss`를 사용하여 Expert를 +1, Policy를 -1로 회귀합니다.

$$ L_{disc} = \frac{1}{2} \Big[ \text{MSE}(D(x_{policy}), -1) + \text{MSE}(D(x_{expert}), 1) \Big] $$
$$ + \lambda_{gp} \Big( \| \nabla_{x_{expert}} D \|^2 + \| \nabla_{x_{policy}} D \|^2 \Big) + \lambda_{logit} \Big( \text{mean}(D(x_{expert})^2) + \text{mean}(D(x_{policy})^2) \Big) $$

*   **Logit 정규화**: 가중치가 아닌 Discriminator의 **출력값(Logit)** 의 제곱합을 억제합니다.

---

## 5. Replay Buffer 운영 방식의 치명적 차이점 (핵심)

**IsaacLab의 구현상 버그(혹은 누락)로 의심되는 매우 중대한 차이입니다.**

망각(Catastrophic Forgetting)을 막기 위해서는 Discriminator를 학습할 때 방금 수집한 최신 Policy 데이터뿐만 아니라 과거 Policy 데이터를 섞어주어야 합니다.

### MimicKit의 완벽한 Replay Buffer 운영
*   `_store_disc_replay_data()`에서 매 스텝 Policy 배치 중 1000개를 버퍼(크기 200,000)에 밀어 넣습니다.
*   `_compute_disc_loss(batch)` 안에서, 현재 배치 사이즈(`N`)와 동일한 수의 샘플을 Replay Buffer에서 뽑아 **현재 데이터와 `torch.cat`으로 합칩니다.**
    ```python
    # mimickit/learning/amp_agent.py - 162번 라인 부근
    replay_data = self._disc_buffer.sample(disc_obs.shape[0]) # N개 샘플링
    replay_obs = replay_data["disc_obs"]
    disc_obs = torch.cat([disc_obs, replay_obs], dim=0) # 현재 N개 + 과거 N개 = 2N개
    norm_disc_obs = self._disc_obs_norm.normalize(disc_obs) # Discriminator 입력
    ```
*   따라서 Discriminator는 항상 절반은 현재 데이터, 절반은 과거 데이터를 보며 엄격하게 판별합니다.

### IsaacLab의 결함 (`update_amp` 함수)
*   `ppo_amp.py` 파일 내에 `add_to_replay_buffer`와 `sample_replay_buffer` 함수가 정의되어 있기는 합니다.
*   **그러나, 정작 Discriminator를 업데이트하는 `update_amp(self, expert_batch, policy_batch)` 함수 내부를 보면 `sample_replay_buffer()`를 호출하는 코드가 아예 없습니다!**
    ```python
    # IsaacLab/rsl_rl/rsl_rl/algorithms/ppo_amp.py - 83번 라인 부근
    def update_amp(self, expert_batch, policy_batch):
        # ... Replay Buffer 샘플링 없이 파라미터로 들어온 policy_batch만 바로 사용함 ...
        expert_logits = self.discriminator.get_logits(expert_batch)
        policy_logits = self.discriminator.get_logits(policy_batch)
    ```
*   이로 인해 IsaacLab의 Discriminator는 매 Iteration마다 **가장 최신의 Policy 데이터에만 적응(Overfitting)**해버립니다. 에이전트가 새로운 동작(꼼수)을 찾으면 Discriminator는 예전의 나쁜 동작을 잊어버리고 오직 그 새로운 동작에만 벌을 주게 되며, 결과적으로 Policy와 Discriminator 간의 균형이 붕괴되어 모션 모방에 실패하게 됩니다.

---

## 🎯 결론 및 조치 제안

현재 IsaacLab에서 학습이 되지 않는 이유는 **1) Pose Termination이 너무 엄격해 탐험을 방해하고 있고, 2) AMP Replay Buffer가 코드상에서 실제로 사용되지 않아 Discriminator가 망각 현상에 빠지기 때문**입니다.

**해결 방안:**
1.  **IsaacLab 환경에서 `pose_termination`을 끄거나 임계값(`pose_termination_dist`)을 매우 크게 늘려주세요.**
2.  **`ppo_amp.py`의 `update_amp` 함수를 수정하여 Replay Buffer에서 샘플링한 데이터를 `policy_batch`에 합쳐서(concat) 학습하도록 코드를 변경해야 합니다.**

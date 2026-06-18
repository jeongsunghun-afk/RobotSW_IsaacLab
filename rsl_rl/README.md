# RSL-RL

A fast and simple implementation of learning algorithms for robotics. For an overview of the library please have a look at https://arxiv.org/pdf/2509.10771.

Environment repositories using the framework:

* **`Isaac Lab`** (built on top of NVIDIA Isaac Sim): https://github.com/isaac-sim/IsaacLab
* **`Legged Gym`** (built on top of NVIDIA Isaac Gym): https://leggedrobotics.github.io/legged_gym/
* **`MuJoCo Playground`** (built on top of MuJoCo MJX and Warp): https://github.com/google-deepmind/mujoco_playground/
* **`mjlab`** (built on top of MuJoCo Warp): https://github.com/mujocolab/mjlab

The library currently supports **PPO** and **Student-Teacher Distillation** with additional features from our research. These include:

* [Random Network Distillation (RND)](https://proceedings.mlr.press/v229/schwarke23a.html) - Encourages exploration by adding
  a curiosity driven intrinsic reward.
* [Symmetry-based Augmentation](https://arxiv.org/abs/2403.04359) - Makes the learned behaviors more symmetrical.

We welcome contributions from the community. Please check our contribution guidelines for more
information.

**Maintainer**: Mayank Mittal and Clemens Schwarke <br/>
**Affiliation**: Robotic Systems Lab, ETH Zurich & NVIDIA <br/>
**Contact**: cschwarke@ethz.ch


## Setup

The package can be installed via PyPI with:

```bash
pip install rsl-rl-lib
```

or by cloning this repository and installing it with:

```bash
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl
pip install -e .
```

The package supports the following logging frameworks which can be configured through `logger`:

* Tensorboard: https://www.tensorflow.org/tensorboard/
* Weights & Biases: https://wandb.ai/site
* Neptune: https://docs.neptune.ai/

For a demo configuration of PPO, please check the [example_config.yaml](config/example_config.yaml) file.


## Contribution Guidelines

For documentation, we adopt the [Google Style Guide](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for docstrings. Please make sure that your code is well-documented and follows the guidelines.

We use the following tools for maintaining code quality:

- [pre-commit](https://pre-commit.com/): Runs a list of formatters and linters over the codebase.
- [ruff](https://github.com/astral-sh/ruff): An extremely fast Python linter and code formatter, written in Rust.

Please check [here](https://pre-commit.com/#install) for instructions to set these up. To run over the entire repository, please execute the following command in the terminal:

```bash
# for installation (only once)
pre-commit install
# for running
pre-commit run --all-files
```

## Citing

If you use this library for your research, please cite the following work:

```text
@article{schwarke2025rslrl,
  title={RSL-RL: A Learning Library for Robotics Research},
  author={Schwarke, Clemens and Mittal, Mayank and Rudin, Nikita and Hoeller, David and Hutter, Marco},
  journal={arXiv preprint arXiv:2509.10771},
  year={2025}
}
```

If you use the library with curiosity-driven exploration (random network distillation), please cite:

```text
@InProceedings{schwarke2023curiosity,
  title = 	 {Curiosity-Driven Learning of Joint Locomotion and Manipulation Tasks},
  author =       {Schwarke, Clemens and Klemm, Victor and Boon, Matthijs van der and Bjelonic, Marko and Hutter, Marco},
  booktitle = 	 {Proceedings of The 7th Conference on Robot Learning},
  pages = 	 {2594--2610},
  year = 	 {2023},
  volume = 	 {229},
  series = 	 {Proceedings of Machine Learning Research},
  publisher =    {PMLR},
  url = 	 {https://proceedings.mlr.press/v229/schwarke23a.html},
}
```

If you use the library with symmetry augmentation, please cite:

```text
@InProceedings{mittal2024symmetry,
  author={Mittal, Mayank and Rudin, Nikita and Klemm, Victor and Allshire, Arthur and Hutter, Marco},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  title={Symmetry Considerations for Learning Task Symmetric Robot Policies},
  year={2024},
  pages={7433-7439},
  doi={10.1109/ICRA57147.2024.10611493}
}
```

---

## Fork 추가 알고리즘 (RGA-Inc/RobotSW_IsaacLab)

이 섹션은 upstream(leggedrobotics/rsl_rl)에는 없는, 이 fork에서 추가한
알고리즘·모듈·러너를 정리합니다.

### 알고리즘 (`rsl_rl/algorithms/`)

| 클래스 | 파일 | 설명 |
|--------|------|------|
| `PPOAMPBase` | `ppo_amp.py` | AMP(Adversarial Motion Priors) 모방학습 알고리즘. `PPO`를 상속하며 `ActorCritic`과 함께 사용 가능한 단순 버전. discriminator 훈련 + 보상 혼합 로직 포함. |
| `PPOAMP` | `ppo_amp.py` | `PPOParkour`를 상속하는 풀 AMP 구현. `ActorCriticRMA`(history/priv encoder)와 함께 사용. LS-GAN / BCE / WGAN 손실 방식 선택 가능. |
| `PPOParkour` | `ppo_parkour.py` | Parkour/RMA 환경 전용 PPO. `ActorCriticRMA`의 multi-branch 구조(proprio + priv_explicit + scan)를 지원하며, RND 탐색 보너스, privileged estimator 공동 학습, 좌우 대칭 데이터 증강, SPO 서러게이트 등 연구 옵션을 포함. |

> `ppo_parkour_original.py`는 레퍼런스 보존용 원본 파일이며 `__init__.py`에 export되지 않습니다.

### 모듈 (`rsl_rl/modules/`)

#### Actor-Critic 아키텍처

| 클래스 | 파일 | 설명 |
|--------|------|------|
| `ActorCriticCNN` | `actor_critic_cnn.py` | CNN 입력을 지원하는 Actor-Critic. 깊이(depth) 이미지 등 픽셀 관측이 있는 환경에 사용. |
| `ActorCriticMoE` | `actor_critic_moe.py` | Dense-softmax Mixture-of-Experts actor를 가진 Actor-Critic. actor head만 MoE로 교체(critic은 단일 MLP). PPO 루프 변경 없음. |
| `ActorCriticRMA` | `actor_critic_parkour.py` | Rapid Motor Adaptation(RMA) 구조. state history encoder(Conv1D) + privileged estimator + scan latent의 multi-branch 입력을 처리. Parkour/고지형 보행 환경의 기본 네트워크. |
| `ActorCriticRMAMoE` | `actor_critic_parkour_moe.py` | `ActorCriticRMA`에 MoE actor head를 결합한 변종. encoder/critic 구조는 그대로 유지. |

#### 보조 모듈

| 클래스 / 모듈 | 파일 | 설명 |
|---------------|------|------|
| `AMPDiscriminator` | `amp_discriminator.py` | AMP용 판별기 네트워크. LS-GAN / BCE / WGAN 보상 방식을 지원하며 empirical 입력 정규화 내장. |
| `Estimator` | `estimator.py` | history 관측에서 privileged 정보(lin_vel 등)를 추정하는 보조 MLP. 배포 시 실제 센서 값 대신 추정값을 사용하는 RMA 패턴에 필수. **`modules/__init__.py`에 export되지 않음** — 러너 내부에서 `from rsl_rl.modules.estimator import Estimator`처럼 직접 import해 사용. |
| `RecurrentDepthBackbone` `StackDepthEncoder` `DepthOnlyFCBackbone58x87` | `depth_backbone.py` | 깊이 이미지를 잠재 벡터로 인코딩하는 백본 모음. GRU 기반 재귀 인코더 / 프레임 스택 인코더 / FC 전용 인코더 세 가지 변종 포함. **`modules/__init__.py`에 export되지 않음** — 러너/네트워크 내부에서 `from rsl_rl.modules.depth_backbone import ...`처럼 직접 import해 사용. |

### 러너 (`rsl_rl/runners/`)

| 클래스 | 파일 | 설명 |
|--------|------|------|
| `OnPolicyRunnerParkour` | `on_policy_runner_parkour.py` | Parkour/RMA 환경용 on-policy 러너. `PPOParkour` + `ActorCriticRMA`를 조합하며 estimator 공동 학습, 관절별 action 통계 로깅, 체크포인트 resume을 지원. |
| `OnPolicyRunnerAMP` `OnPolicyRunnerAMPBase` | `on_policy_runner_amp.py` | AMP 학습 러너. `OnPolicyRunnerParkour`를 상속하고 discriminator 빌드 + AMP 보상 로깅을 추가. |
| `OnPolicyRunnerParkourAMP` | `on_policy_runner_parkour_amp.py` | 지형 AMP(parkour_imitation) 전용 러너. additive + flat_env_mask 기반 AMP 보상 퓨전, spawn curriculum 콜백, discriminator/estimator 체크포인트 저장을 지원. |

> `on_policy_runner_parkour_original.py`는 레퍼런스 보존용 원본 파일이며 `__init__.py`에 export되지 않습니다.

### Upstream 잔존 파일 (수정 없음)

| 파일 / 클래스 | 비고 |
|---------------|------|
| `ppo.py` / `PPO` | upstream 표준 PPO |
| `distillation.py` / `Distillation` | Student-Teacher Distillation |
| `actor_critic.py` / `ActorCritic` | 기본 MLP Actor-Critic |
| `actor_critic_recurrent.py` / `ActorCriticRecurrent` | GRU 기반 재귀 Actor-Critic |
| `student_teacher.py`, `student_teacher_recurrent.py` | 지식 증류 모듈 |
| `rnd.py` / `RandomNetworkDistillation` | 호기심 기반 탐색 (RND) |
| `symmetry.py` / `resolve_symmetry_config` | 대칭 증강 유틸리티 |
| `on_policy_runner.py` / `OnPolicyRunner` | upstream 표준 러너 |

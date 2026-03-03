"""
csv_utils.py
===========
Observation / Action 데이터를 CSV 파일로 저장하는 유틸리티 함수들.
기존 play.py의 save_obs_data_to_csv / save_actions_to_csv 를 분리하였습니다.
"""

import numpy as np
import pandas as pd


def save_obs_data_to_csv(obs_history: list, save_path: str, num_obs: int) -> None:
    """관측 버퍼 히스토리를 CSV 파일로 저장합니다.

    Parameters
    ----------
    obs_history : list[np.ndarray]
        각 타임스텝의 1D 관측 배열 리스트.
    save_path : str
        저장할 CSV 파일 경로.
    num_obs : int
        기대하는 관측 차원 수.
    """
    if not obs_history:
        print("[csv_utils] Warning: obs_history가 비어 있어 CSV 저장을 건너뜁니다.")
        return

    obs_data = np.array(obs_history)
    if obs_data.size == 0:
        print("[csv_utils] Warning: obs 데이터가 비어 있어 CSV 저장을 건너뜁니다.")
        return

    num_timesteps, num_observations = obs_data.shape
    if num_observations != num_obs:
        print(
            f"[csv_utils] Warning: 관측 차원 불일치 "
            f"(데이터={num_observations}, 기대값={num_obs}). 실제 차원 사용."
        )
        num_obs = num_observations

    column_names = [f"obs_{i}" for i in range(num_obs)]
    data_dict = {"Time_Step": np.arange(num_timesteps)}
    for i, col in enumerate(column_names):
        data_dict[col] = obs_data[:, i]

    pd.DataFrame(data_dict).to_csv(save_path, index=False)
    print(f"[csv_utils] 관측 데이터 저장 완료: {save_path}")


def save_actions_to_csv(actions, save_path: str) -> None:
    """단일 타임스텝의 액션 배열을 CSV 파일로 저장합니다.

    Parameters
    ----------
    actions : array-like
        shape (num_envs, action_dim) 또는 (action_dim,).
    save_path : str
        저장할 CSV 파일 경로.
    """
    if actions is None:
        print("[csv_utils] Warning: actions가 None이어서 CSV 저장을 건너뜁니다.")
        return

    actions_np = np.asarray(actions)
    if actions_np.ndim == 1:
        actions_np = actions_np[None, :]
    if actions_np.ndim != 2:
        print(f"[csv_utils] Warning: 예상치 못한 actions 형태 {actions_np.shape}. 건너뜁니다.")
        return

    num_envs, action_dim = actions_np.shape
    records = []
    for env_id in range(num_envs):
        row = {"env_id": env_id}
        for i in range(action_dim):
            row[f"action_{i}"] = actions_np[env_id, i]
        records.append(row)

    pd.DataFrame(records).to_csv(save_path, index=False)
    print(f"[csv_utils] 액션 데이터 저장 완료: {save_path}")

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygame
from IPython import display
from gymnasium import spaces
from sklearn.preprocessing import MinMaxScaler


class RegressionStocksEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, df, render_mode=None, size=5, fps=4):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self._original_df = df

        self._scaler = MinMaxScaler()

        df = pd.DataFrame(self._scaler.fit_transform(df), columns=df.columns)

        #         df = pd.DataFrame(scaler.inverse_transform(target_df), columns=df.columns)

        X = df.iloc[:, :-1]
        self._X = X
        y = df.iloc[:, -1]
        self._y = y

        self._taget_column_name = self._y.name

        self._action_indent = 1e-3
        action_min = y.min() - self._action_indent
        self._action_min = action_min
        action_max = y.max() + self._action_indent
        self._action_max = action_max

        self._initial_taget_location = y[0]

        feature_number = X.shape[1]

        self._dataset = StockDataset(X, y)

        # observing current state of market and news
        describe = X.describe()
        self.observation_space = spaces.Box(
            low=describe.loc['min'].to_numpy(),
            high=describe.loc['max'].to_numpy(),
            shape=(feature_number,), dtype=np.float32
        )

        # prediction of to buy or to sell relatively
        #         self.action_space = spaces.Box(
        #             low=-max_action_bound,
        #             high=+max_action_bound,
        #             dtype=np.float32,
        #         )
        # prediction of close price
        self.action_space = spaces.Box(
            low=action_min,
            high=action_max,
            dtype=np.float32,
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        features, target = self._dataset[self._current_dataset_idx]
        return features

    def _get_info(self):
        return self._current_dataset_idx

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_history = []

        #         self._agent_location = self.np_random.integers(self._obs_min, self._obs_max, size=1, dtype=np.float32)
        self._agent_location = self._initial_taget_location

        self._target_location = self._initial_taget_location

        self._current_dataset_idx = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self._agent_history.append(action)

        terminated = self._current_dataset_idx >= len(self._dataset) - 1

        target = self._y[self._current_dataset_idx]

        reward = 1 / np.abs(action - target)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self._current_dataset_idx += 1
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _inv_transform(self, data):
        scaler = self._scaler
        dummy = pd.DataFrame(np.zeros((len(data), len(self._original_df.columns))), columns=self._original_df.columns)
        dummy[self._taget_column_name] = data
        return scaler.inverse_transform(dummy)

    def _render_frame(self):
        target = self._y[:self._current_dataset_idx]
        agent = self._agent_history

        plt.figure(3)
        plt.clf()
        plt.cla()
        plt.plot(target, label='target')
        plt.plot(agent, label='agent')

        plt.title("%s | Step: %d" % (env, self._current_dataset_idx))
        plt.ylim((target.min() - 0, target.max() + 0))
        plt.legend()
        display.clear_output(wait=True)
        display.display(plt.gcf())

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

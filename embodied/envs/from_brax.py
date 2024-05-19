import functools

import embodied
import gym
import numpy as np
import jax
from gym import spaces
from gym.vector import utils
from brax.envs.base import PipelineEnv
from brax.io import image
from typing import Optional


class FromBrax(embodied.Env):

    def __init__(self, env: PipelineEnv, obs_key='image', act_key='action', seed: int = 0,
                 backend: Optional[str] = None, vector_env: int = 0, **kwargs):
        self._env = env
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 1 / self._env.dt
        }
        self.backend = backend
        self.seed(seed)
        self._state = None
        self.num_envs = vector_env

        raw_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        if self.num_envs > 0:
            self.observation_space = utils.batch_space(raw_space, self.num_envs)

            action = jax.tree_map(np.array, self._env.sys.actuator.ctrl_range)
            action_space = spaces.Box(action[:, 0], action[:, 1], dtype='float32')
            self.action_space = utils.batch_space(action_space, self.num_envs)
        else:
            self.observation_space = raw_space

            action = jax.tree_map(np.array, self._env.sys.actuator.ctrl_range)
            self.action_space = spaces.Box(action[:, 0], action[:, 1], dtype='float32')

        self._obs_dict = hasattr(self.observation_space, 'spaces')
        self._act_dict = hasattr(self.action_space, 'spaces')

        self._obs_key = obs_key
        self._act_key = act_key
        self._done = True
        self._info = None

        def reset(key):
            key1, key2 = jax.random.split(key)
            state = self._env.reset(key2)
            return state, state.obs, key1

        self._reset = jax.jit(reset, backend=self.backend)

        def step(state, action):
            state = self._env.step(state, action)
            info = {**state.metrics, **state.info}
            return state, state.obs, state.reward, state.done, info

        self.__step = jax.jit(step, backend=self.backend)

    @property
    def env(self):
        return self._env

    @property
    def info(self):
        return self._info

    def reset(self):
        self._state, obs, self._key = self._reset(self._key)
        return obs

    def _step(self, action):
        self._state, obs, reward, done, info = self.__step(self._state, action)
        return obs, reward, done, info

    def seed(self, seed: int = 0):
        self._key = jax.random.key(seed)

    @functools.cached_property
    def obs_space(self):
        return {
            'image': embodied.Space(np.uint8, (64, 64, 3)),
            'reward': embodied.Space(np.float32),
            'is_first': embodied.Space(bool),
            'is_last': embodied.Space(bool),
            'is_terminal': embodied.Space(bool),
        }

    @functools.cached_property
    def act_space(self):
        if self._act_dict:
            spaces = self._flatten(self.action_space.spaces)
        else:
            spaces = {self._act_key: self.action_space}
        spaces = {k: self._convert(v) for k, v in spaces.items()}
        spaces['reset'] = embodied.Space(bool)
        return spaces

    def step(self, action):
        if action['reset'] or self._done:
            self._done = False
            obs = self.reset()
            return self._obs(obs, 0.0, is_first=True)
        if self._act_dict:
            action = self._unflatten(action)
        else:
            action = action[self._act_key]
        obs, reward, self._done, self._info = self._step(action)
        return self._obs(
            obs, reward,
            is_last=bool(self._done),
            is_terminal=bool(self._info.get('is_terminal', self._done)))

    def _obs(
            self, obs, reward, is_first=False, is_last=False, is_terminal=False):
        if not self._obs_dict:
            obs = {self._obs_key: obs}
        obs = self._flatten(obs)
        obs = {k: np.asarray(v) for k, v in obs.items()}
        obs.update(
            image=self.render(),
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal)
        return obs

    def _render(self, mode='rgb_array', width=64, height=64):
        if mode == 'rgb_array':
            sys, state = self._env.sys, self._state
            if state is None:
                raise RuntimeError('must call reset or step before rendering')
            return image.render_array(sys, state.pipeline_state, height, width)
        else:
            return super().render(mode=mode)

    def render(self):
        image = self._render('rgb_array')
        assert image is not None
        return image

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

    def _flatten(self, nest, prefix=None):
        result = {}
        for key, value in nest.items():
            key = prefix + '/' + key if prefix else key
            if isinstance(value, gym.spaces.Dict):
                value = value.spaces
            if isinstance(value, dict):
                result.update(self._flatten(value, key))
            else:
                result[key] = value
        return result

    def _unflatten(self, flat):
        result = {}
        for key, value in flat.items():
            parts = key.split('/')
            node = result
            for part in parts[:-1]:
                if part not in node:
                    node[part] = {}
                node = node[part]
            node[parts[-1]] = value
        return result

    def _convert(self, space):
        if hasattr(space, 'n'):
            return embodied.Space(np.int32, (), 0, space.n)
        return embodied.Space(space.dtype, space.shape, space.low, space.high)

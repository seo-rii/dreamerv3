import functools

import embodied
import gym
import numpy as np
import jax
from gym import spaces
from brax.envs.base import PipelineEnv
from typing import Optional


class FromBrax(embodied.Env):

    def __init__(self, env: PipelineEnv, obs_key='image', act_key='action', seed: int = 0,
                 backend: Optional[str] = None, vector_env: int = 1, **kwargs):
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

        self.observation_space = raw_space

        action = jax.tree_map(np.array, self._env.sys.actuator.ctrl_range)
        self.action_space = spaces.Box(action[:, 0], action[:, 1], dtype='float32')

        self._obs_dict = hasattr(self.observation_space, 'spaces')
        self._act_dict = hasattr(self.action_space, 'spaces')

        self._obs_key = obs_key
        self._act_key = act_key
        self._done = np.zeros(vector_env, dtype=bool)
        self._info = None

        def reset(key):
            key1, *keys = jax.random.split(key, num=self.num_envs + 1)
            keys = jax.numpy.array(keys)
            states = jax.vmap(lambda x: self._env.reset(x))(keys)
            return states, jax.vmap(lambda x: x.obs)(states), key1

        self._reset = jax.jit(reset, backend=self.backend)

        def step(states, action):
            states = jax.vmap(lambda x, y: self._env.step(x, y))(states, action)
            info = jax.vmap(lambda x: {**x.metrics, **x.info})(states)
            return states, jax.vmap(lambda x: x.obs)(states), jax.vmap(lambda x: x.reward)(states), \
                jax.vmap(lambda x: x.done)(states), info

        self.__step = jax.jit(step, backend=self.backend)

    @property
    def env(self):
        return self._env

    @property
    def info(self):
        return self._info

    def reset(self):
        self._state, obs, self._key = self._reset(self._key)
        return self.render()

    def _step(self, action):
        self._state, obs, reward, done, info = self.__step(self._state, action)
        return self.render(), reward, done, info

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
        reset_mask = action['reset'] | self._done.astype('bool')
        if np.any(reset_mask):
            self._done[reset_mask] = False
            obs = self.reset()
            return self._obs(obs, np.zeros(self.num_envs), is_first=reset_mask)

        if self._act_dict:
            action = self._unflatten(action)
        else:
            action = action[self._act_key]

        obs, reward, self._done, self._info = self._step(action)

        is_last = np.array(self._done, dtype=bool)
        is_terminal = np.array([self._info.get(i, {}).get('is_terminal', d) for i, d in enumerate(self._done)],
                               dtype=bool)

        return self._obs(obs, reward, is_last=is_last, is_terminal=is_terminal)

    def _obs(
            self, obs, reward, is_first=None, is_last=None, is_terminal=None):
        if not self._obs_dict:
            obs = {self._obs_key: obs}
        obs = self._flatten(obs)
        obs = {k: np.asarray(v) for k, v in obs.items()}
        obs.update(
            reward=np.asarray(reward, dtype=np.float32),
            is_first=is_first if is_first is not None else np.zeros(self.num_envs, dtype=bool),
            is_last=is_last if is_last is not None else np.zeros(self.num_envs, dtype=bool),
            is_terminal=is_terminal if is_terminal is not None else np.zeros(self.num_envs, dtype=bool))
        return obs

    def _render(self, mode='rgb_array', width=64, height=64):
        if mode == 'rgb_array':
            sys, states = self._env.sys, self._state
            if states is None:
                raise RuntimeError('must call reset or step before rendering')

            class VState:
                def __init__(self, q, qd):
                    self.q = q
                    self.qd = qd

            q = states.pipeline_state.q
            qd = states.pipeline_state.qd
            states = [VState(q[i], qd[i]) for i in range(self.num_envs)]

            return jax.numpy.array(self._env.render(states, width=width, height=height))
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

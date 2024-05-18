import jax
from jax import numpy as jnp
from typing import Tuple

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import html, mjcf

from etils import epath
import mujoco
import tempfile
import webbrowser
import mediapy


class Walker2d(PipelineEnv):
    def __init__(
            self,
            forward_reward_weight: float = 1.0,
            ctrl_cost_weight: float = 1e-3,
            healthy_reward: float = 1.0,
            terminate_when_unhealthy: bool = True,
            healthy_z_range: Tuple[float, float] = (0.8, 2.0),
            healthy_angle_range=(-1.0, 1.0),
            reset_noise_scale=5e-3,
            exclude_current_positions_from_observation=True,
            **kwargs
    ):

        # path = epath.resource_path('brax') / 'envs/assets/walker2d.xml'
        path = epath.Path('new_walker.xml')
        mj_model = mujoco.MjModel.from_xml_path((path).as_posix())
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6
        sys = mjcf.load_model(mj_model)
        n_frames = 4
        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)
        kwargs['backend'] = 'mjx'
        super().__init__(sys=sys, **kwargs)

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

    def reset(self, rng: jnp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(
            rng2, (self.sys.qd_size(),), minval=low, maxval=hi
        )

        pipeline_state = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(pipeline_state)
        reward, done, zero = jnp.zeros(3)
        metrics = {
            'reward_forward': zero,
            'reward_ctrl': zero,
            'reward_healthy': zero,
            'x_position': zero,
            'x_velocity': zero,
        }
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        x_velocity = (
                             pipeline_state.x.pos[0, 0] - pipeline_state0.x.pos[0, 0]
                     ) / self.dt
        forward_reward = self._forward_reward_weight * x_velocity

        z, angle = pipeline_state.x.pos[0, 2], pipeline_state.q[2]
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range
        is_healthy = (
                (z > min_z) & (z < max_z) * (angle > min_angle) & (angle < max_angle)
        )
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))

        obs = self._get_obs(pipeline_state)
        reward = forward_reward + healthy_reward - ctrl_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        state.metrics.update(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_healthy=healthy_reward,
            x_position=pipeline_state.x.pos[0, 0],
            x_velocity=x_velocity,
        )

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: base.State) -> jnp.ndarray:
        """Returns the environment observations."""
        position = pipeline_state.q
        position = position.at[1].set(pipeline_state.x.pos[0, 2])
        velocity = jnp.clip(pipeline_state.qd, -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        return jnp.concatenate((position, velocity))


# example format
# envs.register_environment('walker2d', Walker2d)


if __name__ == '__main__':
    env = Walker2d()
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    state = jit_env_reset(jax.random.key(0))
    states = []

    for i in range(600):
        print(f'step {i}')
        action = jax.random.uniform(jax.random.key(i), (env.action_size,))
        state = jit_env_step(state, action)
        states.append(state)

    mediapy.show_video(env.render([state.pipeline_state for state in states], width=64, height=64)[0])

    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
        url = 'file://' + f.name
        f.write(html.render(env.sys, [state.pipeline_state for state in states]))
        webbrowser.open(url)

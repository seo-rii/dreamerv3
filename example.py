import warnings
from functools import partial as bind

import dreamerv3
import embodied

warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')


def main():
    # import os
    # os.environ["MUJOCO_GL"] = "egl"
    # os.environ["MUJOCO_RENDERER"] = "egl"
    config = embodied.Config(dreamerv3.Agent.configs['defaults'])
    config = config.update({
        **dreamerv3.Agent.configs['size12m'],
        'logdir': f'~/logdir/{embodied.timestamp()}-example',
        'run.train_ratio': 32,
    })
    config = embodied.Flags(config).parse()
    try:
        vector_env = config.vector_env
    except AttributeError:
        vector_env = 1
        config = config.update({'vector_env': vector_env})

    print('Logdir:', config.logdir)
    logdir = embodied.Path(config.logdir)
    logdir.mkdir()
    config.save(logdir / 'config.yaml')

    def make_agent(config):
        env = make_env(config)
        # remove first dimension by env_batch to get single env obs_space

        obs_space = env.obs_space
        act_space = env.act_space

        agent = dreamerv3.Agent(obs_space, act_space, config)
        env.close()
        return agent

    def make_logger(config):
        logdir = embodied.Path(config.logdir)
        return embodied.Logger(embodied.Counter(), [
            embodied.logger.TerminalOutput(config.filter),
            embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
            embodied.logger.TensorBoardOutput(logdir),
            # embodied.logger.WandbOutput(logdir.name, config=config),
        ])

    def make_replay(config):
        return embodied.replay.Replay(
            length=config.batch_length,
            capacity=config.replay.size,
            directory=embodied.Path(config.logdir) / 'replay',
            online=config.replay.online)

    def make_env(config, env_id=0):
        from suite.cheetah import Cheetah
        from embodied.envs import from_brax

        env = Cheetah(batch_size=max(1, config.vector_env))
        env = from_brax.FromBrax(env, vector_env=config.vector_env)

        env = dreamerv3.wrap_env(env, config)
        return env

    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        batch_length_eval=config.batch_length_eval,
        replay_context=config.replay_context,
        vector_env=config.vector_env,
    )

    embodied.run.train(
        bind(make_agent, config),
        bind(make_replay, config),
        bind(make_env, config),
        bind(make_logger, config), args)


if __name__ == '__main__':
    main()

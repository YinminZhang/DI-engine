from typing import Union, Optional, List, Any, Tuple
import os
import torch
import logging
from functools import partial
from tensorboardX import SummaryWriter
import numpy as np

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, SampleCollector, BaseSerialEvaluator, BaseSerialCommander, create_buffer, \
    create_serial_collector
from ding.config import read_config, compile_config
from ding.policy import create_policy, PolicyFactory
from ding.utils import set_pkg_seed
from ding.utils import save_data


def serial_pipeline_data_generation(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        max_iterations: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline entry.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - max_iterations (:obj:`Optional[torch.nn.Module]`): Learner's max iteration. Pipeline will stop \
            when reaching this iteration.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = input_cfg
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    env_fn = None if env_setting is None else env_setting[0]
    cfg = compile_config(cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True)
    # Create main components: env, policy
    if env_setting is None:
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    else:
        env_fn, collector_env_cfg, evaluator_env_cfg = env_setting
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval', 'command'])

    # Create worker components: learner, collector, evaluator, replay buffer, commander.
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collect_demo_policy = policy.collect_mode
    # collect_demo_policy = policy.collect_function(
    #     policy._forward_eval,
    #     policy._process_transition,
    #     policy._get_train_sample,
    #     policy._reset_eval,
    #     policy._get_attribute,
    #     policy._set_attribute,
    #     policy._state_dict_eval,
    #     policy._load_state_dict_eval,
    # )
    collector = create_serial_collector(
        cfg.policy.collect.collector,
        env=collector_env,
        policy=collect_demo_policy,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name
    )
    evaluator = BaseSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    replay_buffer = create_buffer(cfg.policy.other.replay_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name)
    commander = BaseSerialCommander(
        cfg.policy.other.commander, learner, collector, evaluator, replay_buffer, policy.command_mode
    )
    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    # import ipdb;ipdb.set_trace()
    learner.call_hook('before_run')

    collect_kwargs = commander.step()
    new_data = collector.collect(n_sample=cfg.policy.other.replay_buffer.replay_buffer_size, policy_kwargs=collect_kwargs)
    new_data, new_trajs = reshuffle_dataset(new_data, cfg.env.collector_env_num, full_episode = cfg.policy.generate.full_episode)
    compute_episode(new_trajs, cfg.env.collector_env_num, visualization = {
        'flag':cfg.policy.generate.visualization, 
        'title':cfg.policy.generate.vis_title,
        'vis_path':cfg.policy.generate.vis_path,
    })
    replay_buffer.push(new_data, cur_collector_envstep=0)

    # Save replay buffer data

    save_data(replay_buffer, cfg.policy.learn.save_path)
    # state_dict = torch.load(cfg.policy.learn.hook.load_ckpt_before_run, map_location='cpu')
    # policy.load_state_dict(state_dict)
    stop, reward = evaluator.eval(train_iter= learner.train_iter, envstep= collector.envstep)
    # Learner's after_run hook.
    # learner.call_hook('after_run')

    return policy


def reshuffle_dataset(data: List, collect_env: int, full_episode = False) -> List:
    indices = np.repeat(np.arange(collect_env), (len(data)//collect_env)).tolist() + \
              np.arange(len(data)%collect_env).tolist()
    base_indices = (collect_env * np.tile(np.arange(len(data)//collect_env), collect_env)).tolist() + \
                   np.array([len(data)//collect_env * collect_env]).repeat(len(data)%collect_env).tolist()
    print(np.array(indices)+np.array(base_indices))
    new_data = [data[i] for i in (np.array(indices)+np.array(base_indices)).tolist()]
    new_trajs={i: new_data[i * len(new_data)//collect_env:(i+1) * len(new_data)//collect_env] for i in range(collect_env)}
    if full_episode:
        trajs = {i: new_data[i * len(new_data)//collect_env:(i+1) * len(new_data)//collect_env] for i in range(collect_env)}
        new_data = []
        new_trajs = {}
        for env_ids, transitions in trajs.items():
            done_index = 0
            for i in range(len(transitions)-1, -1, -1):
                if transitions[i]['done']:
                    done_index = i
                    break
            new_data.extend(transitions[:done_index+1])
            new_trajs[env_ids]=transitions[:done_index+1]
    return new_data, new_trajs




def compute_episode(new_trajs: dict, collect_env: int, visualization: dict = {'flag': False, 'title':"", 'vis_path': ""}) -> List:

    # trajs = {i: data[i * len(data)//collect_env:(i+1) * len(data)//collect_env] for i in range(collect_env)}

    episode_info = []
    for env_ids, transitions in new_trajs.items():
        lens = 0
        sum_reward = 0
        for transition in transitions:
            sum_reward += transition['reward']
            lens+=1
            if transition['done']:
                episode_info.append({'rew': sum_reward.item(), 'done': True, 'length': lens, 'env': env_ids})
                sum_reward = 0
                lens = 0
        if sum_reward:
            episode_info.append({'rew': sum_reward.item(), 'done': False, 'length': lens, 'env': env_ids})
            # sum_reward = 0

    done_episode = [i['rew'] for i in episode_info if i['done']]
    print(f"episode numbers: {len(episode_info)}, "
          f"done episode numbers: {len(done_episode)},"
          f"done_avg_reward: {sum(done_episode)/len(done_episode)}")
    if visualization['flag']:
        reward_analysis(episode_info, visualization['title'], visualization['vis_path'])


def reward_analysis(episode_info: dict, title: str, path: str):
    import matplotlib.pyplot as plt
    import seaborn as sns
    rewards = []
    lens = []
    for d in episode_info:
        if d['done']:
            rewards.append(d['rew'])
            lens.append(d['length'])

    rewards = np.array(rewards)
    lens = np.array(lens)
    print(f'avg rews: {rewards.mean()}, avg lengths: {lens.mean()}')


    hfont = {'family':'Times New Roman', 'size': '15'}
    fig, ax = plt.subplots()

    plt.grid(linestyle='--', alpha=0.5)
    ax.set_ylabel('Numbers of path', hfont)
    ax.set_xlabel('Episode rewards', hfont)
    plt.title(title, hfont)
    # plt.text(0.04, 0.8, f'max reward: {int(rewards.max())}\n\
    plt.text(0.74, 0.8, f'max reward: {int(rewards.max())}\n\
min reward: {int(rewards.min())}\n\
avg reward: {int(rewards.mean())}\n\
avg length: {int(lens.mean())}', transform=ax.transAxes)

    plt.hist(rewards, bins=50)
    plt.savefig(path)
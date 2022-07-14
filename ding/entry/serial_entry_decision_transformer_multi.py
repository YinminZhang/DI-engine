"""
The code is adapted from https://github.com/nikhilbarhate99/min-decision-transformer
"""
from typing import Union, Optional, List, Any, Tuple
from functools import partial
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, create_buffer
from ding.config import read_config, compile_config
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.utils.data import create_dataset
import os
import logging
import random
import time
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from ding.rl_utils import discount_cumsum
from ding.utils.data.dataset import D4RLTrajectoryDataset


def serial_pipeline_dt_multi(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        max_train_iter: Optional[int] = int(5e2),
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
        - max_train_iter (:obj:`Optional[int]`): Maximum policy update iterations in training.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = input_cfg
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)

    # Dataset
    traj_dataset = D4RLTrajectoryDataset(cfg)
    traj_data_loader = DataLoader(
        traj_dataset, batch_size=cfg.policy.batch_size, shuffle=True, pin_memory=True, drop_last=True
    )
    # get state stats from dataset
    state_mean, state_std = traj_dataset.get_state_stats()

    cfg.policy.learn.dataset_path = cfg.policy.learn.sub_dataset_path
    traj_auxiliary_dataset = D4RLTrajectoryDataset(cfg)
    traj_auxiliary_data_loader = DataLoader(
        traj_auxiliary_dataset, batch_size=cfg.policy.batch_size, shuffle=True, pin_memory=True, drop_last=True
    )
    # get state stats from dataset
    state_auxiliary_mean, state_auxiliary_std = traj_auxiliary_dataset.get_state_stats()
    # # Env, Policy
    # env_fn, _, evaluator_env_cfg = get_vec_env_setting(cfg.env, collect=False)
    # evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    # # Random seed
    # evaluator_env.seed(cfg.seed, dynamic_seed=False)

    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'eval'])

    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    # evaluator = InteractionSerialEvaluator(
    #     cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    # )

    # evaluator.eval(learner.save_checkpoint, learner.train_iter)
    # total_update_times = 0
    # stop, eval_reward = policy.evaluate(total_update_times, state_mean, state_std)

    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook('before_run')
    stop = False
    total_update_times = 0
    for i in range(max_train_iter):
        import ipdb; ipdb.set_trace()
        iterator = enumerate(traj_data_loader)
        iterator_auxiliary = enumerate(traj_auxiliary_data_loader)
        for j in range(len(traj_data_loader)):
            # for k, train_data in next(iterator):
            k, train_data = next(iterator)
            # if evaluator.should_eval(learner.train_iter):
            #     stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter)
            #     if stop:
            #         break
            # import ipdb; ipdb.set_trace()
            learner.train(train_data)
            total_update_times += 1
            if total_update_times != 0 and (total_update_times + 1) % 1000 == 0:
                stop, eval_reward = policy.evaluate(total_update_times, state_mean, state_std)
                tb_logger.add_scalar('iter/evaluate_reward', eval_reward, total_update_times)
                # stop, eval_auxiliary_reward = policy.evaluate(total_update_times, state_auxiliary_mean, state_auxiliary_std)
                # tb_logger.add_scalar('iter/evaluate_auxiliary_reward', eval_auxiliary_reward, total_update_times)
                if stop:
                    break
            # for k, train_data in next(iterator_auxiliary):
            k, train_data = next(iterator_auxiliary)
            # if evaluator.should_eval(learner.train_iter):
            #     stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter)
            #     if stop:
            #         break
            learner.train(train_data)
            total_update_times += 1
            if total_update_times != 0 and total_update_times % 1000 == 0:
                # stop, eval_reward = policy.evaluate(total_update_times, state_mean, state_std)
                # tb_logger.add_scalar('iter/evaluate_reward', eval_reward, total_update_times)
                stop, eval_auxiliary_reward = policy.evaluate(total_update_times, state_auxiliary_mean, state_auxiliary_std)
                tb_logger.add_scalar('iter/evaluate_auxiliary_reward', eval_auxiliary_reward, total_update_times)
                if stop:
                    break
        if stop:
            break
        if total_update_times >= 1e5:
            break
    learner.call_hook('after_run')
    return policy, stop

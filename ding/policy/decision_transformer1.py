"""The code is adapted from https://github.com/nikhilbarhate99/min-decision-transformer
"""

from cmath import e
from random import random
from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
from torch.distributions import Normal, Independent
from ding.torch_utils import Adam, to_device
from ditk import logging
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample, \
    qrdqn_nstep_td_data, qrdqn_nstep_td_error, get_nstep_return_data
from ding.model import model_wrap
from ding.utils.data.dataset import D4RLTrajectoryDataset
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from datetime import datetime
from ding.torch_utils import one_hot
import numpy as np
import torch.nn.functional as F
import torch
import gym
import copy
import os
import csv
from .dqn import DQNPolicy


@POLICY_REGISTRY.register('dt1')
class DT1Policy(DQNPolicy):
    r"""
    Overview:
        Policy class of DT algorithm in discrete environments.
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='dt',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        priority=False,
        # (float) Reward's future discount factor, aka. gamma.
        discount_factor=0.97,
        # (int) N-step reward for target q_value estimation
        nstep=1,
        obs_shape=4,
        action_shape=2,
        # encoder_hidden_size_list=[128, 128, 64],
        dataset='medium',  # medium / medium-replay / medium-expert
        rtg_scale=1000,  # normalize returns to go
        max_eval_ep_len=1000,  # max len of one episode
        num_eval_ep=10,  # num of evaluation episodes
        batch_size=64,  # training batch size
        wt_decay=1e-4,
        warmup_steps=10000,
        max_train_iters=200,
        context_len=20,
        n_blocks=3,
        embed_dim=128,
        dropout_p=0.1,
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # batch_size=64,
            learning_rate=1e-4,
            # ==============================================================
            # The following configs are algorithm-specific
            # ==============================================================
        ),
        # collect_mode config
        collect=dict(),
        eval=dict(),
        # other config
        other=dict(),
    )

    def _init_learn(self) -> None:
        r"""
            Overview:
                Learn mode init method. Called by ``self.__init__``.
                Init the optimizer, algorithm config, main and target models.
            """

        self.stop_value = self._cfg.stop_value
        self.env_name = self._cfg.env_name
        dataset = self._cfg.dataset  # medium / medium-replay / medium-expert
        # rtg_scale: scale of `return to go`
        # rtg_target: max target of `return to go`
        # Our goal is normalize `return to go` to (0, 1), which will favour the covergence.
        # As a result, we usually set rtg_scale == rtg_target.
        self.rtg_scale = self._cfg.rtg_scale    # normalize returns to go
        self.rtg_target = self._cfg.rtg_target  # max target reward_to_go
        self.max_eval_ep_len = self._cfg.max_eval_ep_len  # max len of one episode
        self.num_eval_ep = self._cfg.num_eval_ep  # num of evaluation episodes

        lr = self._cfg.learn.learning_rate  # learning rate
        wt_decay = self._cfg.wt_decay  # weight decay
        warmup_steps = self._cfg.warmup_steps  # warmup steps for lr scheduler

        max_train_iters = self._cfg.max_train_iters

        self.context_len = self._cfg.context_len  # K in decision transformer
        n_blocks = self._cfg.n_blocks  # num of transformer blocks
        embed_dim = self._cfg.embed_dim  # embedding (hidden) dim of transformer
        dropout_p = self._cfg.dropout_p  # dropout probability

        # # load data from this file
        # dataset_path = f'{self._cfg.dataset_dir}/{env_d4rl_name}.pkl'

        # saves model and csv in this directory
        self.log_dir = self._cfg.log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # training and evaluation device
        self.device = torch.device(self._device)

        self.start_time = datetime.now().replace(microsecond=0)
        self.start_time_str = self.start_time.strftime("%y-%m-%d-%H-%M-%S")

        # prefix = "dt_" + env_d4rl_name
        self.prefix = "dt_" + self.env_name if isinstance(self.env_name, str) else "dt_" + '-'.join(self.env_name)

        save_model_name = self.prefix + "_model_" + self.start_time_str + ".pt"
        self.save_model_path = os.path.join(self.log_dir, save_model_name)
        self.save_best_model_path = self.save_model_path[:-3] + "_best.pt"

        log_csv_name = self.prefix + "_log_" + self.start_time_str + ".csv"
        log_csv_path = os.path.join(self.log_dir, log_csv_name)

        self.csv_writer = csv.writer(open(log_csv_path, 'a', 1))
        csv_header = (["duration", "num_updates", "eval_avg_reward", "eval_avg_ep_len", "eval_d4rl_score"])

        self.csv_writer.writerow(csv_header)

        dataset_path = self._cfg.learn.dataset_path
        logging.info("=" * 60)
        logging.info("start time: " + self.start_time_str)
        logging.info("=" * 60)

        logging.info("device set to: " + str(self.device))
        logging.info("dataset path: " + dataset_path)
        logging.info("model save path: " + self.save_model_path)
        logging.info("log csv save path: " + log_csv_path)

        if isinstance(self.env_name, list):
            self._env = [gym.make(env_name) for env_name in self.env_name]
        else:
            if 'minigrid' in self.env_name.lower():
                from easydict import EasyDict
                from dizoo.minigrid.envs import MiniGridEnv
                config = dict(
                    env_id=self.env_name,
                    flat_obs=True,
                    max_step=300,
                )
                self._env = MiniGridEnv(EasyDict(config))
            elif 'smac' in self.env_name.lower():
                from easydict import EasyDict
                from dizoo.smac.envs import SMACEnv
                self._env = SMACEnv(EasyDict(self._cfg.env))
            else:
                self._env = gym.make(self.env_name)

        self.state_dim = self._cfg.model.state_dim
        self.act_dim = self._cfg.model.act_dim

        self._learn_model = self._model
        self._optimizer = torch.optim.AdamW(self._learn_model.parameters(), lr=lr, weight_decay=wt_decay)

        self._scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._optimizer, lambda steps: min((steps + 1) / warmup_steps, 1)
        )

        self.max_env_score = -1.0

        self._with_decrement_return = self._cfg.learn.get('with_decrement_return', True)
        self.discrete = self._cfg.get('discrete', False)
        self.discrete_bin = self._cfg.get('discrete_bin', 0)
        self.inverse_discrete_bin = self._cfg.get('inverse_discrete_bin', False)
        if self._cfg.model.get('state_goal', False):
            self.ratio = self._cfg.get('expert_state_ratio', 1.0)

    def _forward_learn(self, data: list) -> Dict[str, Any]:
        r"""
            Overview:
                Forward and backward function of learn mode.
            Arguments:
                - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs']
            Returns:
                - info_dict (:obj:`Dict[str, Any]`): Including current lr and loss.
        """
        self._learn_model.train()
        timesteps, states, actions, returns_to_go, traj_mask = data
        timesteps = timesteps.to(self.device)  # B x T
        states = states.to(self.device)  # B x T x state_dim
        actions = actions.to(self.device)  # B x T x act_dim
        returns_to_go = returns_to_go.to(self.device).float()  # B x T x 1
        traj_mask = traj_mask.to(self.device)  # B x T
        action_target = torch.clone(actions).detach().to(self.device)

        if self.inverse_discrete_bin and self.discrete:
            # returns_to_go = self._cfg.discrete_bin - returns_to_go
            returns_to_go = ((returns_to_go) - (self._cfg.discrete_bin//2)) % self._cfg.discrete_bin

        # import ipdb; ipdb.set_trace()
        if isinstance(self._env, list):
            idx = [e.observation_space.shape for e in self._env].index((states.shape[-1],))
            env = self._env[idx]
            act_dim = self.act_dim[idx]
            state_dim = self.state_dim[idx]
        else:
            act_dim = self.act_dim
            state_dim = self.state_dim
        # The shape of `returns_to_go` may differ with different dataset (B x T or B x T x 1),
        # and we need a 3-dim tensor
        if len(returns_to_go.shape) == 2:
            returns_to_go = returns_to_go.unsqueeze(-1)
        # if discrete
        if not self._cfg.model.continuous:
            actions = one_hot(actions.squeeze(-1), num=act_dim)

        if self._cfg.get('random_mask_ratio', 0.0):
            # import ipdb; ipdb.set_trace()
            B, T, C = actions.shape
            states_inputs = torch.clone(states).detach().to(self.device)
            actions_inputs = torch.clone(actions).detach().to(self.device)
            returns_to_go_inputs = torch.clone(returns_to_go).detach().to(self.device)
            index = np.random.choice(T*3, int(self._cfg.random_mask_ratio*T*3), replace=False)
            index_return = index[index<T]
            index_states = index[(index<2*T) & (index>T)] - T
            index_action = index[index>2*T] - 2 * T
            actions_inputs[:, index_action, :] = torch.rand_like(actions_inputs[:, index_action, :]).to(self.device)
            states_inputs[:, index_states, :] = torch.rand_like(states_inputs[:, index_states, :]).to(self.device)
            returns_to_go_inputs[:, index_return, :] = torch.rand_like(returns_to_go_inputs[:, index_return, :]).to(self.device)
        else:
            states_inputs = torch.clone(states).detach().to(self.device)
            actions_inputs = torch.clone(actions).detach().to(self.device)
            returns_to_go_inputs = torch.clone(returns_to_go).detach().to(self.device)
        state_preds, action_preds, return_preds = self._learn_model.forward(
            timesteps=timesteps, states=states_inputs, actions=actions_inputs, returns_to_go=returns_to_go_inputs
        )

        traj_mask = traj_mask.view(-1, )

        # only consider non padded elements
        action_preds = action_preds.view(-1, act_dim)[traj_mask > 0]

        if self._cfg.model.continuous:
            action_target = action_target.view(-1, act_dim)[traj_mask > 0]
        else:
            action_target = action_target.view(-1)[traj_mask > 0]

        if self._cfg.model.continuous:
            action_loss = F.mse_loss(action_preds, action_target)
        else:
            action_loss = F.cross_entropy(action_preds, action_target)

        if self._cfg.get('auxiliary_loss', False):
            # import ipdb; ipdb.set_trace()
            B, T, C = state_preds.shape
            # only consider non padded elements
            state_preds = state_preds.view(-1, state_dim)[traj_mask > 0]
            return_preds = return_preds.view(-1, 1)[traj_mask > 0]
            states_target = torch.clone(states).detach().to(self.device).view(-1, state_dim)[traj_mask > 0]
            returns_to_go_target = torch.clone(returns_to_go).detach().to(self.device).view(-1, 1)[traj_mask > 0]

            states_loss = F.mse_loss(state_preds.reshape(B, T, C)[:, :-1, :], 
                                    states_target.reshape(B, T, C)[:, 1:, :])
            returns_to_go_loss = F.mse_loss(return_preds.reshape(B, T, 1)[:, :-1, :], 
                                            returns_to_go_target.reshape(B, T, 1)[:, 1:, :])
        else:
            states_loss, returns_to_go_loss = 0, 0

        total_loss = action_loss + states_loss + returns_to_go_loss 
        self._optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._learn_model.parameters(), 0.25)
        self._optimizer.step()
        self._scheduler.step()
           

        if self._cfg.get('auxiliary_loss', False):
            return {
                'cur_lr': self._optimizer.state_dict()['param_groups'][0]['lr'],
                'total_loss': total_loss.detach().cpu().item(),
                'action_loss': action_loss.detach().cpu().item(),
                'states_loss': states_loss.detach().cpu().item(),
                'returns_to_go_loss': returns_to_go_loss.detach().cpu().item(),
            }
        else:
            return {
                'cur_lr': self._optimizer.state_dict()['param_groups'][0]['lr'],
                'total_loss': total_loss.detach().cpu().item(),
                'action_loss': action_loss.detach().cpu().item(),
                'states_loss': torch.zeros_like(action_loss).item(),
                'returns_to_go_loss': torch.zeros_like(action_loss).item(),
            }

    def evaluate_on_env(self, state_mean=None, state_std=None, render=False):
        
        # import ipdb; ipdb.set_trace()
        if isinstance(self._env, list):
            idx = [e.observation_space.shape for e in self._env].index(state_mean.shape)
            env = self._env[idx]
            act_dim = self.act_dim[idx]
            state_dim = self.state_dim[idx]
        else:
            env = self._env
            act_dim = self.act_dim
            state_dim = self.state_dim

        eval_batch_size = 1  # required for forward pass

        results = {}
        total_reward = 0
        total_timesteps = 0

        # state_dim = env.observation_space.shape[0]
        # act_dim = env.action_space.shape[0]

        if state_mean is None:
            self.state_mean = torch.zeros((state_dim, )).to(self.device)
        else:
            self.state_mean = torch.from_numpy(state_mean).to(self.device)

        if state_std is None:
            self.state_std = torch.ones((state_dim, )).to(self.device)
        else:
            self.state_std = torch.from_numpy(state_std).to(self.device)

        # same as timesteps used for training the transformer
        # also, crashes if device is passed to arange()
        timesteps = torch.arange(start=0, end=self.max_eval_ep_len, step=1)
        timesteps = timesteps.repeat(eval_batch_size, 1).to(self.device)

        self._learn_model.eval()

        with torch.no_grad():

            for _ in range(self.num_eval_ep):

                # zeros place holders
                # continuous action
                actions = torch.zeros(
                    (eval_batch_size, self.max_eval_ep_len, act_dim), dtype=torch.float32, device=self.device
                )

                # discrete action # TODO
                # actions = torch.randint(0,self.act_dim,[eval_batch_size, self.max_eval_ep_len, 1],
                # dtype=torch.long, device=self.device)

                states = torch.zeros(
                    (eval_batch_size, self.max_eval_ep_len, state_dim), dtype=torch.float32, device=self.device
                )
                if self._cfg.model.get('state_goal', False):
                    rewards_to_go = torch.zeros(
                        (eval_batch_size, self.max_eval_ep_len, self.state_dim), dtype=torch.float32, device=self.device
                    )
                    # import ipdb; ipdb.set_trace()
                    if self.env_name == 'Hopper-v3':
                        ratio = self.ratio
                        medium = torch.tensor([1.3112685680389404, -0.08469261974096298, -0.5382542610168457, -0.07201051712036133, 0.049344852566719055, 2.106581926345825, -0.15014714002609253, 0.008785112760961056, -0.2848515808582306, -0.18539157509803772, -0.2846825420856476], device=self.device).float()
                        expert = torch.tensor([1.348850965499878, -0.11204933375120163, -0.550708532333374, -0.1316361278295517, -0.0031308038160204887, 2.606743574142456, 0.022303320467472076, -0.01657019928097725, -0.06813174486160278, -0.05349138379096985, 0.04012826085090637], device=self.device).float()
                        self.rtg_target = ratio * expert + (1 - ratio) * medium
                    elif self.env_name == 'HalfCheetah-v3':
                        ratio = self.ratio
                        medium = torch.tensor([-0.0684533417224884, 0.016389088705182076, -0.18353942036628723, -0.2762347161769867, -0.3406224250793457, -0.09342234581708908, -0.21320895850658417, -0.08775322139263153, 5.17288875579834, -0.042751897126436234, -0.03614707291126251, 0.14031197130680084, 0.06066203862428665, 0.09548277407884598, 0.06728935986757278, 0.005867088679224253, 0.013621795922517776], device=self.device).float()
                        expert = torch.tensor([-0.04489380493760109, 0.03229331225156784, 0.06037571281194687, -0.1708163619041443, -0.1948034167289734, -0.05755220726132393, 0.09699936211109161, 0.03238246962428093, 11.04673957824707, -0.08001191914081573, -0.3236390948295593, 0.36369049549102783, 0.42418915033340454, 0.40823298692703247, 1.1076111793518066, -0.4877481460571289, -0.07378903776407242], device=self.device).float()
                        self.rtg_target = ratio * expert + (1 - ratio) * medium
                    elif self.env_name == 'Walker2d-v3':
                        ratio = self.ratio
                        medium = torch.tensor([1.2189686298370361, 0.14164122939109802, -0.03705529123544693, -0.13813918828964233, 0.5138940811157227, -0.047185178846120834, -0.47281956672668457, 0.042268332093954086, 2.3946220874786377, -0.03145574405789375, 0.044610973447561264, -0.024010702967643738, -0.10130638629198074, 0.09077298641204834, -0.004206471145153046, -0.12143220007419586, -0.5497634410858154], device=self.device).float()
                        expert = torch.tensor([1.2385023832321167, 0.19576017558574677, -0.10479946434497833, -0.1859603375196457, 0.22955934703350067, 0.022799348458647728, -0.3737868070602417, 0.33816614747047424, 3.9246129989624023, -0.00483204610645771, 0.025160973891615868, -0.005045710131525993, -0.017238497734069824, -0.4814542531967163, 0.0004888453986495733, -0.0007198172388598323, 0.0035148432943969965], device=self.device).float()
                        self.rtg_target = ratio * expert + (1 - ratio) * medium
                    self.rtg_scale = 1.0
                    # import ipdb; ipdb.set_trace()
                    if self._cfg.get('top_state', 0):
                        import pickle
                        import numpy as np
                        env_id = self.env_name.split('-')[0].lower()
                        dataset_path = f'/mnt/lustre/zhangyinmin.p/dataset/d4rl_data/{env_id}-expert-v2.pkl'

                        with open(dataset_path, 'rb') as f:
                            trajectories = pickle.load(f)
                        rews = []
                        stats = []
                        for i in range(len(trajectories)):
                            traj = trajectories[i]
                            for j in range(len(traj['rewards']) - 20):
                                rews.append(sum(traj['rewards'][j:j+20]))
                                stats.append(np.array(traj['observations'][j:j+20]).mean(axis=0))
                        rews = np.array(rews)
                        stats = np.array(stats)
                        idx = np.argsort(rews)[::-1][:int(rews.shape[0]*self._cfg.top_state)]
                        # import ipdb; ipdb.set_trace()
                        # print(stats[idx, :].mean(axis=0))
                        self.rtg_target = torch.tensor(stats[idx, :].mean(axis=0), device=self.device).float()
                    # self.rtg_target = expert
                    if self._cfg.get('use_past_reward_expert', False):
                        import pickle
                        import numpy as np
                        env_id = self.env_name.split('-')[0].lower()
                        dataset_path = f'/mnt/lustre/zhangyinmin.p/dataset/d4rl_data/{env_id}-expert-v2.pkl'

                        with open(dataset_path, 'rb') as f:
                            trajectories = pickle.load(f)
                        # rews = []
                        episode_states = []
                        for i in range(len(trajectories)):
                            traj = trajectories[i]
                            episode_states.append(traj['observations'])
                        episode_states = np.array(episode_states)
                        self.rtg_target = torch.tensor(episode_states.mean(axis=0), device=self.device).float()
                else:    
                    rewards_to_go = torch.zeros(
                        (eval_batch_size, self.max_eval_ep_len, 1), dtype=torch.float32, device=self.device
                    )
                if self._cfg.get('use_past_reward', False):
                    self.rtg_target = 0
                    rewards_to_go = torch.zeros(
                        (eval_batch_size, self.max_eval_ep_len, 1), dtype=torch.float32, device=self.device
                    )

                # init episode
                running_state = env.reset()
                running_reward = 0
                running_rtg = self.rtg_target / self.rtg_scale

                for t in range(self.max_eval_ep_len):

                    total_timesteps += 1

                    # add state in placeholder and normalize
                    states[0, t] = torch.from_numpy(running_state).to(self.device)
                    # states[0, t] = (states[0, t].cpu() - self.state_mean.cpu().numpy()) / self.state_std.cpu().numpy()
                    states[0, t] = (states[0, t] - self.state_mean) / self.state_std

                    # calcualate running rtg and add it in placeholder
                    if self._with_decrement_return:
                        running_rtg = running_rtg - (running_reward / self.rtg_scale)
                    else:
                        running_rtg = running_rtg
                    if self._cfg.get('use_past_reward', False):
                        running_rtg = running_rtg + (running_reward / self.rtg_scale)
                    if self.discrete:
                        if self._with_decrement_return:
                            running_rtg = running_rtg - (running_reward / (self.rtg_target * self.rtg_scale))
                        else:
                            running_rtg = self.discrete_bin
                        if self.inverse_discrete_bin:
                            # running_rtg = 0
                            running_rtg = (self.discrete_bin - self.discrete_bin//2)%self.discrete_bin

                    rewards_to_go[0, t] = running_rtg
                    # import ipdb;ipdb.set_trace()
                    if t < self.context_len:
                        _, act_preds, _ = self._learn_model.forward(
                            timesteps[:, :self.context_len], states[:, :self.context_len],
                            actions[:, :self.context_len], rewards_to_go[:, :self.context_len]
                        )
                        act = act_preds[0, t].detach()
                    else:
                        _, act_preds, _ = self._learn_model.forward(
                            timesteps[:, t - self.context_len + 1:t + 1], states[:, t - self.context_len + 1:t + 1],
                            actions[:, t - self.context_len + 1:t + 1], rewards_to_go[:, t - self.context_len + 1:t + 1]
                        )
                        act = act_preds[0, -1].detach()

                    # if discrete
                    if not self._cfg.model.continuous:
                        act = torch.argmax(act)
                    running_state, running_reward, done, _ = env.step(act.cpu().numpy())

                    # add action in placeholder
                    actions[0, t] = act

                    total_reward += running_reward

                    if render:
                        env.render()
                    if done:
                        break

        results['eval/avg_reward'] = total_reward / self.num_eval_ep
        results['eval/avg_ep_len'] = total_timesteps / self.num_eval_ep

        return results

    def evaluate(self, total_update_times, state_mean=None, state_std=None, render=False):
        results = self.evaluate_on_env(state_mean, state_std, render)

        eval_avg_reward = results['eval/avg_reward']
        eval_avg_ep_len = results['eval/avg_ep_len']
        if isinstance(self._env, list):
            idx = [e.observation_space.shape for e in self._env].index(state_mean.shape)
            env_name = self.env_name[idx]
        else:
            env_name = self.env_name
        time_elapsed = str(datetime.now().replace(microsecond=0) - self.start_time)
        env_key = env_name.split('-')[0].lower()
        if env_key in D4RLTrajectoryDataset.REF_MAX_SCORE:
            eval_d4rl_score = self.get_d4rl_normalized_score(results['eval/avg_reward'], env_name) * 100
            log_str = (
                "=" * 60 + '\n' + "time elapsed: " + time_elapsed + '\n' + "num of updates: " + str(total_update_times) +
                '\n' + '\n' + "eval avg reward: " + format(eval_avg_reward, ".5f") + '\n' + "eval avg ep len: " +
                format(eval_avg_ep_len, ".5f") + '\n' + "eval d4rl score: " + format(eval_d4rl_score, ".5f")
            )
            log_data = [time_elapsed, total_update_times, eval_avg_reward, eval_avg_ep_len, eval_d4rl_score]

        else:
            log_str = (
                "=" * 60 + '\n' + "time elapsed: " + time_elapsed + '\n' + "num of updates: " + str(total_update_times) +
                '\n' + '\n' + "eval avg reward: " + format(eval_avg_reward, ".5f") + '\n' + "eval avg ep len: " +
                format(eval_avg_ep_len, ".5f") + '\n'
            )
            log_data = [time_elapsed, total_update_times, eval_avg_reward, eval_avg_ep_len]
        logging.info(log_str)

        log_csv_name = self.prefix + "_log_" + self.start_time_str + ".csv"
        log_csv_path = os.path.join(self.log_dir, log_csv_name)

        self.csv_writer.writerow(log_data)

        # save model
        logging.info("eval_avg_reward: " + format(eval_avg_reward, ".5f"))
        eval_env_score = eval_avg_reward
        if eval_env_score >= self.max_env_score:
            logging.info("saving max env score model at: " + self.save_best_model_path)
            # torch.save(self._learn_model.state_dict(), self.save_best_model_path)
            self.max_env_score = eval_env_score

        logging.info("saving current model at: " + self.save_model_path)
        # torch.save(self._learn_model.state_dict(), self.save_model_path)

        return self.max_env_score >= self.stop_value, eval_env_score

    def get_d4rl_normalized_score(self, score, env_name):
        env_key = env_name.split('-')[0].lower()
        assert env_key in D4RLTrajectoryDataset.REF_MAX_SCORE, \
            f'no reference score for {env_key} env to calculate d4rl score'
        d4rl_max_score, d4rl_min_score = D4RLTrajectoryDataset.REF_MAX_SCORE, D4RLTrajectoryDataset.REF_MIN_SCORE
        return (score - d4rl_min_score[env_key]) / (d4rl_max_score[env_key] - d4rl_min_score[env_key])

    def _state_dict_learn(self) -> Dict[str, Any]:
        return {
            'model': self._learn_model.state_dict(),
            # 'target_model': self._target_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._learn_model.load_state_dict(state_dict['model'])
        # self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def default_model(self) -> Tuple[str, List[str]]:
        return 'dt', ['ding.model.template.decision_transformer']

    def _monitor_vars_learn(self) -> List[str]:
        return ['cur_lr', 
                'action_loss',
                'total_loss',
                'states_loss',
                'returns_to_go_loss',]

    def set_norm_statistics(self, mean: float, std: float) -> None:
        r"""
        Overview:
            Set (mean, std) for state normalization.
        Arguments:
            - mean (:obj:`float`): Float type data, the mean of state in offlineRL dataset.
            - std (:obj:`float`): Float type data, the std of state in offlineRL dataset.
        Returns:
            - None
        """
        self._mean = mean
        self._std = std
    
    def _forward_eval(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of eval mode, similar to ``self._forward_collect``.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): The dict of predicting action for the interaction with env.
        ReturnsKeys
            - necessary: ``action``
            - optional: ``logit``
        """
        # import ipdb; ipdb.set_trace()
        data_id = list(data.keys())
        data = default_collate(list(data.values()))

        eval_batch_size = 1  # required for forward pass
        # same as timesteps used for training the transformer
        # also, crashes if device is passed to arange()
        timesteps = torch.arange(start=0, end=self.max_eval_ep_len, step=1)
        timesteps = timesteps.repeat(eval_batch_size, 1).to(self.device)

        timesteps, states, actions, returns_to_go, traj_mask = data
        if self.cfg.collect.normalize_states:
            states = (states - self._mean) / self._std
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(timesteps=timesteps, states=states, actions=actions, returns_to_go=returns_to_go)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import torch
import copy
import numpy as np

from ding.torch_utils import Adam, to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('ddpg')
class DDPGPolicy(Policy):
    r"""
    Overview:
        Policy class of DDPG algorithm.
    Property:
        learn_mode, collect_mode, eval_mode

    Config:

        == ====================  ========    =============  =================================   =======================
        ID Symbol                Type        Default Value  Description                         Other(Shape)
        == ====================  ========    =============  =================================   =======================
        1  ``type``              str         ddpg           | RL policy register name, refer    | this arg is optional,
                                                            | to registry ``POLICY_REGISTRY``   | a placeholder
        2  ``cuda``              bool        True           | Whether to use cuda for network   |
        3  | ``random_``         int         25000          | Number of randomly collected      | Default to 25000 for
           | ``collect_size``                               | training samples in replay        | DDPG/TD3, 10000 for
           |                                                | buffer when training starts.      | sac.
        4  | ``model.twin_``     bool        False          | Whether to use two critic         | Default False for
           | ``critic``                                     | networks or only one.             | DDPG, Clipped Double
           |                                                |                                   | Q-learning method in
           |                                                |                                   | TD3 paper.
        5  | ``learn.learning``  float       1e-3           | Learning rate for actor           |
           | ``_rate_actor``                                | network(aka. policy).             |
        6  | ``learn.learning``  float       1e-3           | Learning rates for critic         |
           | ``_rate_critic``                               | network (aka. Q-network).         |
        7  | ``learn.actor_``    int         2              | When critic network updates       | Default 1 for DDPG,
           | ``update_freq``                                | once, how many times will actor   | 2 for TD3. Delayed
           |                                                | network update.                   | Policy Updates method
           |                                                |                                   | in TD3 paper.
        8  | ``learn.noise``     bool        False          | Whether to add noise on target    | Default False for
           |                                                | network's action.                 | DDPG, True for TD3.
           |                                                |                                   | Target Policy Smoo-
           |                                                |                                   | thing Regularization
           |                                                |                                   | in TD3 paper.
        9  | ``learn.-``         bool        False          | Determine whether to ignore       | Use ignore_done only
           | ``ignore_done``                                | done flag.                        | in halfcheetah env.
        10 | ``learn.-``         float       0.005          | Used for soft update of the       | aka. Interpolation
           | ``target_theta``                               | target network.                   | factor in polyak aver
           |                                                |                                   | aging for target
           |                                                |                                   | networks.
        11 | ``collect.-``       float       0.1            | Used for add noise during co-     | Sample noise from dis
           | ``noise_sigma``                                | llection, through controlling     | tribution, Ornstein-
           |                                                | the sigma of distribution         | Uhlenbeck process in
           |                                                |                                   | DDPG paper, Guassian
           |                                                |                                   | process in ours.
        == ====================  ========    =============  =================================   =======================
    """

    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='ddpg',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool type) on_policy: Determine whether on-policy or off-policy.
        # on-policy setting influences the behaviour of buffer.
        # Default False in DDPG.
        on_policy=False,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        # Default False in DDPG.
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (int) Number of training samples(randomly collected) in replay buffer when training starts.
        # Default 25000 in DDPG/TD3.
        random_collect_size=25000,
        model=dict(
            # (bool) Whether to use two critic networks or only one.
            # Clipped Double Q-Learning for Actor-Critic in original TD3 paper.
            # Default False for DDPG, True for TD3.
            twin_critic=False,
        ),
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=1,
            # (int) Minibatch size for gradient descent.
            batch_size=256,
            # Learning rates for actor network(aka. policy).
            learning_rate_actor=1e-3,
            # Learning rates for critic network(aka. Q-network).
            learning_rate_critic=1e-3,
            # (bool) Whether ignore done(usually for max step termination env. e.g. pendulum)
            # Note: Gym wraps the MuJoCo envs by default with TimeLimit environment wrappers.
            # These limit HalfCheetah, and several other MuJoCo envs, to max length of 1000.
            # However, interaction with HalfCheetah always gets done with done is False,
            # Since we inplace done==True with done==False to keep
            # TD-error accurate computation(``gamma * (1 - done) * next_v + reward``),
            # when the episode step is greater than max episode step.
            ignore_done=False,
            # (float type) target_theta: Used for soft update of the target network,
            # aka. Interpolation factor in polyak averaging for target networks.
            # Default to 0.005.
            target_theta=0.005,
            # (float) discount factor for the discounted sum of rewards, aka. gamma.
            discount_factor=0.99,
            # (int) When critic network updates once, how many times will actor network update.
            # Delayed Policy Updates in original TD3 paper.
            # Default 1 for DDPG, 2 for TD3.
            actor_update_freq=1,
            # (bool) Whether to add noise on target network's action.
            # Target Policy Smoothing Regularization in original TD3 paper.
            # Default False for DDPG, True for TD3.
            noise=False,
        ),
        collect=dict(
            # (int) Only one of [n_sample, n_episode] shoule be set
            n_sample=1,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
            # It is a must to add noise during collection. So here omits "noise" and only set "noise_sigma".
            noise_sigma=0.1,
        ),
        eval=dict(evaluator=dict(eval_freq=1000, ), ),
        other=dict(
            replay_buffer=dict(
                # (int) Maximum size of replay buffer.
                replay_buffer_size=1000000,
            ),
        ),
    )

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init actor and critic optimizers, algorithm config, main and target models.
        """
        # add cql
        self._data_type = self._cfg.learn.data_type
        self._data_path = self._cfg.learn.data_path
        self.min_q_version = 3
        self.temp = 1.
        self.min_q_weight = self._cfg.learn.min_q_weight
        self.with_lagrange = True
        self.lagrange_thresh = self._cfg.learn.lagrange_thresh
        if self.with_lagrange:
            self.target_action_gap = self.lagrange_thresh
            self.log_alpha_prime = torch.tensor(0.).to(self._device).requires_grad_()
            self.alpha_prime_optimizer = Adam(
                [self.log_alpha_prime],
                lr=self._cfg.learn.learning_rate_q,
            )


        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        # actor and critic optimizer
        self._optimizer_actor = Adam(
            self._model.actor.parameters(),
            lr=self._cfg.learn.learning_rate_actor,
        )
        self._optimizer_critic = Adam(
            self._model.critic.parameters(),
            lr=self._cfg.learn.learning_rate_critic,
        )
        self._use_reward_batch_norm = self._cfg.get('use_reward_batch_norm', False)

        self._gamma = self._cfg.learn.discount_factor
        self._actor_update_freq = self._cfg.learn.actor_update_freq
        self._twin_critic = self._cfg.model.twin_critic  # True for TD3, False for DDPG

        # main and target models
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.learn.target_theta}
        )
        if self._cfg.learn.noise:
            self._target_model = model_wrap(
                self._target_model,
                wrapper_name='action_noise',
                noise_type='gauss',
                noise_kwargs={
                    'mu': 0.0,
                    'sigma': self._cfg.learn.noise_sigma
                },
                noise_range=self._cfg.learn.noise_range
            )
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()
        self._target_model.reset()

        self._forward_learn_cnt = 0  # count iterations

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including at least actor and critic lr, different losses.
        """
        loss_dict = {}
        data = default_preprocess_learn(
            data,
            use_priority=self._cfg.priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=False
        )
        if self._cuda:
            data = to_device(data, self._device)
        # ====================
        # critic learn forward
        # ====================
        self._learn_model.train()
        self._target_model.train()
        obs = data.get('obs')
        next_obs = data.get('next_obs')
        reward = data.get('reward')
        if self._use_reward_batch_norm:
            reward = (reward - reward.mean()) / (reward.std() + 1e-8)
        # current q value
        q_value = self._learn_model.forward(data, mode='compute_critic')['q_value']
        q_value_dict = {}
        if self._twin_critic:
            q_value_dict['q_value'] = q_value[0].mean()
            q_value_dict['q_value_twin'] = q_value[1].mean()
        else:
            q_value_dict['q_value'] = q_value.mean()
        # target q value. SARSA: first predict next action, then calculate next q value
        with torch.no_grad():
            next_action = self._target_model.forward(next_obs, mode='compute_actor')['action']
            next_data = {'obs': next_obs, 'action': next_action}
            target_q_value = self._target_model.forward(next_data, mode='compute_critic')['q_value']
        if self._twin_critic:
            # TD3: two critic networks
            target_q_value = torch.min(target_q_value[0], target_q_value[1])  # find min one as target q value
            # network1
            td_data = v_1step_td_data(q_value[0], target_q_value, reward, data['done'], data['weight'])
            critic_loss, td_error_per_sample1 = v_1step_td_error(td_data, self._gamma)
            loss_dict['critic_loss'] = critic_loss
            # network2(twin network)
            td_data_twin = v_1step_td_data(q_value[1], target_q_value, reward, data['done'], data['weight'])
            critic_twin_loss, td_error_per_sample2 = v_1step_td_error(td_data_twin, self._gamma)
            loss_dict['critic_twin_loss'] = critic_twin_loss
            td_error_per_sample = (td_error_per_sample1 + td_error_per_sample2) / 2
        else:
            # DDPG: single critic network
            td_data = v_1step_td_data(q_value, target_q_value, reward, data['done'], data['weight'])
            critic_loss, td_error_per_sample = v_1step_td_error(td_data, self._gamma)
            loss_dict['critic_loss'] = critic_loss

        # add CQL
        curr_actions_tensor = self._get_policy_actions(data, self._num_actions)
        new_curr_actions_tensor = self._get_policy_actions({'obs': next_obs}, self._num_actions)

        # random_actions_tensor = torch.FloatTensor(q2_pred.shape[0] * self.num_random, actions.shape[-1]).uniform_(-1, 1) # .cuda()
        random_actions_tensor = torch.FloatTensor(curr_actions_tensor.shape).uniform_(-1, 1).to(
            curr_actions_tensor.device)

        obs_repeat = obs.unsqueeze(1).repeat(1, self._num_actions, 1).view(obs.shape[0] *
                                                                           self._num_actions, obs.shape[1])
        act_repeat = data['action'].unsqueeze(1).repeat(1, self._num_actions, 1).view(data['action'].shape[0] *
                                                                                      self._num_actions,
                                                                                      data['action'].shape[1])
        q_pred = self._get_q_value({'obs': obs_repeat, 'action': act_repeat})
        q_rand = self._get_q_value({'obs': obs_repeat, 'action': random_actions_tensor})
        # q2_rand = self._get_q_value(obs, random_actions_tensor, network=self.qf2)
        q_curr_actions = self._get_q_value({'obs': obs_repeat, 'action': curr_actions_tensor})
        # q2_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.qf2)
        q_next_actions = self._get_q_value({'obs': obs_repeat, 'action': new_curr_actions_tensor})
        # q2_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.qf2)
        cat_q1 = torch.stack(
            [q_rand[0], q_pred[0], q_next_actions[0], q_curr_actions[0]], 1
        )
        cat_q2 = torch.stack(
            [q_rand[1], q_pred[1], q_next_actions[1], q_curr_actions[1]], 1
        )
        std_q1 = torch.std(cat_q1, dim=1)
        std_q2 = torch.std(cat_q2, dim=1)
        if self.min_q_version == 3:
            # importance sammpled version
            random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
            # import ipdb
            # ipdb.set_trace()
            cat_q1 = torch.stack(
                [q_rand[0] - random_density, q_next_actions[0],
                 q_curr_actions[0]], 1
            )
            cat_q2 = torch.stack(
                [q_rand[1] - random_density, q_next_actions[1],
                 q_curr_actions[1]], 1
            )

        min_qf1_loss = torch.logsumexp(cat_q1 / self.temp, dim=1, ).mean() * self.min_q_weight * self.temp
        if self._twin_critic:
            min_qf2_loss = torch.logsumexp(cat_q2 / self.temp, dim=1, ).mean() * self.min_q_weight * self.temp

        """Subtract the log likelihood of data"""
        min_qf1_loss = min_qf1_loss - q_pred[0].mean() * self.min_q_weight
        if self._twin_critic:
            min_qf2_loss = min_qf2_loss - q_pred[1].mean() * self.min_q_weight

        if self.with_lagrange:
            alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
            min_qf1_loss = alpha_prime * (min_qf1_loss - self.target_action_gap)
            if self._twin_critic:
                min_qf2_loss = alpha_prime * (min_qf2_loss - self.target_action_gap)

            self.alpha_prime_optimizer.zero_grad()
            if self._twin_critic:
                alpha_prime_loss = (-min_qf1_loss - min_qf2_loss) * 0.5
            else:
                alpha_prime_loss = -min_qf1_loss
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()

        loss_dict['critic_loss'] += min_qf1_loss
        if self._twin_critic:
            loss_dict['twin_critic_loss'] += min_qf2_loss

        # ================
        # critic update
        # ================
        self._optimizer_critic.zero_grad()
        for k in loss_dict:
            if 'critic' in k:
                loss_dict[k].backward()
        self._optimizer_critic.step()
        # ===============================
        # actor learn forward and update
        # ===============================
        # actor updates every ``self._actor_update_freq`` iters
        if (self._forward_learn_cnt + 1) % self._actor_update_freq == 0:
            actor_data = self._learn_model.forward(data['obs'], mode='compute_actor')
            actor_data['obs'] = data['obs']
            if self._twin_critic:
                actor_loss = -self._learn_model.forward(actor_data, mode='compute_critic')['q_value'][0].mean()
            else:
                actor_loss = -self._learn_model.forward(actor_data, mode='compute_critic')['q_value'].mean()

            loss_dict['actor_loss'] = actor_loss
            # actor update
            self._optimizer_actor.zero_grad()
            actor_loss.backward()
            self._optimizer_actor.step()
        # =============
        # after update
        # =============
        loss_dict['total_loss'] = sum(loss_dict.values())
        self._forward_learn_cnt += 1
        self._target_model.update(self._learn_model.state_dict())
        return {
            'cur_lr_actor': self._optimizer_actor.defaults['lr'],
            'cur_lr_critic': self._optimizer_critic.defaults['lr'],
            # 'q_value': np.array(q_value).mean(),
            'action': data.get('action').mean(),
            'priority': td_error_per_sample.abs().tolist(),
            **loss_dict,
            **q_value_dict,
        }

    def _state_dict_learn(self) -> Dict[str, Any]:
        return {
            'model': self._learn_model.state_dict(),
            'optimizer_actor': self._optimizer_actor.state_dict(),
            'optimizer_critic': self._optimizer_critic.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._learn_model.load_state_dict(state_dict['model'])
        self._optimizer_actor.load_state_dict(state_dict['optimizer_actor'])
        self._optimizer_critic.load_state_dict(state_dict['optimizer_critic'])

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, collect model.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        # collect model
        self._collect_model = model_wrap(
            self._model,
            wrapper_name='action_noise',
            noise_type='gauss',
            noise_kwargs={
                'mu': 0.0,
                'sigma': self._cfg.collect.noise_sigma
            },
            noise_range=None
        )
        self._collect_model.reset()

    def _forward_collect(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of collect mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - output (:obj:`dict`): Dict type data, including at least inferred action according to input obs.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, mode='compute_actor')
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> Dict[str, Any]:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - model_output (:obj:`dict`): Output of collect model, including at least ['action']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \
                (here 'obs' indicates obs after env step, i.e. next_obs).
        Return:
            - transition (:obj:`Dict[str, Any]`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': model_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        return get_train_sample(data, self._unroll_len)

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval model. Unlike learn and collect model, eval model does not need noise.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='base')
        self._eval_model.reset()

    def _forward_eval(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of collect mode, similar to ``self._forward_collect``.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - output (:obj:`dict`): Dict type data, including at least inferred action according to input obs.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data, mode='compute_actor')
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def default_model(self) -> Tuple[str, List[str]]:
        return 'qac', ['ding.model.template.qac']

    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return variables' name if variables are to used in monitor.
        Returns:
            - vars (:obj:`List[str]`): Variables' name list.
        """
        ret = [
            'cur_lr_actor', 'cur_lr_critic', 'critic_loss', 'actor_loss', 'total_loss', 'q_value', 'q_value_twin',
            'action'
        ]
        if self._twin_critic:
            ret += ['critic_twin_loss']
        return ret

    def _get_policy_actions(self, data: Dict, num_actions=10):

        # evaluate to get action distribution
        obs = data['obs']
        obs = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
        output = self._collect_model.forward(obs, mode='compute_actor')

        return output['pred']

    def _get_q_value(self, data: Dict, keep=True):

        new_q_value = self._learn_model.forward(data, mode='compute_critic')['q_value']
        if self._twin_critic and not keep:
            new_q_value = torch.min(new_q_value[0], new_q_value[1])

        return new_q_value
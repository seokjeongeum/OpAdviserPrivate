import os
import math
import pdb

import torch
import pickle
import logging
import numpy as np
import torch.nn as nn
from torch.nn import init, Parameter
import torch.nn.functional as F
import torch.optim as optimizer
from torch.autograd import Variable

from .ouprocess import OUProcess
#from .replay_memory import ReplayMemory
from .prioritized_replay_memory import PrioritizedReplayMemory
# from autotune.knobs import logger


# code from https://github.com/Kaixhin/NoisyNet-A3C/blob/master/model.py
class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.05, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=True)  # TODO: Adapt for no bias
        # µ^w and µ^b reuse self.weight and self.bias
        self.sigma_init = sigma_init
        self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))  # σ^w
        self.sigma_bias = Parameter(torch.Tensor(out_features))  # σ^b
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        self.register_buffer('epsilon_bias', torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'sigma_weight'):  # Only init after all params added (otherwise super().__init__() fails)
            init.uniform(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            init.uniform(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            init.constant(self.sigma_weight, self.sigma_init)
            init.constant(self.sigma_bias, self.sigma_init)

    def forward(self, input):
        return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight), self.bias + self.sigma_bias * Variable(self.epsilon_bias))

    def sample_noise(self):
        self.epsilon_weight = torch.randn(self.out_features, self.in_features)
        self.epsilon_bias = torch.randn(self.out_features)

    def remove_noise(self):
        self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
        self.epsilon_bias = torch.zeros(self.out_features)


class Normalizer(object):

    def __init__(self, mean, variance):
        if isinstance(mean, list):
            mean = np.array(mean)
        if isinstance(variance, list):
            variance = np.array(variance)
        self.mean = mean
        self.std = np.sqrt(variance+0.00001)

    def normalize(self, x):
        if isinstance(x, list):
            x = np.array(x)
        x = x - self.mean
        x = x / self.std
        return Variable(torch.FloatTensor(x))

    def __call__(self, x, *args, **kwargs):
        return self.normalize(x)


import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (seq_len, batch_size, embed_dim)
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attended))
        feedforward = self.ff(x)
        return self.norm2(x + self.dropout(feedforward))

# Modify the Actor class to include Transformer
class Actor(nn.Module):

    def __init__(self, n_states, n_actions, noisy=False,transformer=True):
        super(Actor, self).__init__()
        
        # Initial embedding layer
        self.embed = nn.Linear(n_states, 128)
        
        # Add transformer block
        self.transformer = TransformerBlock(
            embed_dim=128,
            num_heads=4,
            ff_dim=256
        )
        
        self.layers = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(64),
            nn.Linear(64, n_actions)
        )

        self.act = nn.Sigmoid()
        self._init_weights()
        self.use_transformer=transformer

    def _init_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 1e-2)
                m.bias.data.uniform_(-0.1, 0.1)


    def forward(self, states):
        # Embed the states
        x = self.embed(states)
        # Add sequence dimension for transformer
        x = x.unsqueeze(0)  # shape: (1, batch_size, embed_dim)
        if self.use_transformer:
            # Apply transformer
            x = self.transformer(x)
            # Remove sequence dimension
            x = x.squeeze(0)
        # Apply remaining layers
        actions = self.act(self.layers(x))
        return actions


class Critic(nn.Module):

    def __init__(self, n_states, n_actions,transformer):
        super(Critic, self).__init__()
        self.act = nn.Tanh()
        
        # State pathway
        self.state_input = nn.Linear(n_states, 128)
        self.state_transformer = TransformerBlock(
            embed_dim=128,
            num_heads=4,
            ff_dim=256
        )
        
        # Action pathway
        self.action_input = nn.Linear(n_actions, 128)
        self.action_transformer = TransformerBlock(
            embed_dim=128,
            num_heads=4,
            ff_dim=256
        )
        
        # Combined layers
        self.layers = nn.Sequential(
            nn.Linear(128 * 2, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1)
        )
        self._init_weights()
        self.use_transformer=transformer

    def _init_weights(self):
        self.state_input.weight.data.normal_(0.0, 1e-2)
        self.state_input.bias.data.uniform_(-0.1, 0.1)

        self.action_input.weight.data.normal_(0.0, 1e-2)
        self.action_input.bias.data.uniform_(-0.1, 0.1)

        for m in self.layers:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 1e-2)
                m.bias.data.uniform_(-0.1, 0.1)

    def forward(self, states, actions):
        # Process states
        states = self.act(self.state_input(states))
        states = states.unsqueeze(0)
        if self.use_transformer:
            states = self.state_transformer(states)
            states = states.squeeze(0)

        # Process actions
        actions = self.act(self.action_input(actions))
        actions = actions.unsqueeze(0)
        if self.use_transformer:
            actions = self.action_transformer(actions)
            actions = actions.squeeze(0)

        # Combine and process through remaining layers
        _input = torch.cat([states, actions], dim=1)
        value = self.layers(_input)
        return value


class DDPG(object):

    def __init__(self, n_states, n_actions, opt, mean=None, var=None, ouprocess=True, supervised=False, debug=False,transformer=True):
        """ DDPG Algorithms
        Args:
            n_states: int, dimension of states
            n_actions: int, dimension of actions
            opt: dict, params
            supervised, bool, pre-train the actor with supervised learning
        """
        self.n_states = n_states
        self.n_actions = n_actions

        # Params
        self.alr = opt['alr']
        self.clr = opt['clr']
        self.model_name = opt['model']
        self.batch_size = opt['batch_size']
        self.gamma = opt['gamma']
        self.tau = opt['tau']
        self.ouprocess = ouprocess
        self.debug=debug
        self.logger = logging.getLogger(self.__class__.__name__)

        if mean is None:
            mean = np.zeros(n_states)
        if var is None:
            var = np.zeros(n_states)

        self.normalizer = Normalizer(mean, var)
        self.transformer=transformer

        if supervised:
            self._build_actor()
            self.logger.info("Supervised Learning Initialized")
        else:
            # Build Network
            self._build_network()
            self.logger.info('Finish Initializing Networks')

        self.replay_memory = PrioritizedReplayMemory(capacity=opt['memory_size'])
        # self.replay_memory = ReplayMemory(capacity=opt['memory_size'])
        self.noise = OUProcess(n_actions)
        self.debug = debug
        self.logger.info('DDPG Initialzed!')

    @staticmethod
    def totensor(x):
        return Variable(torch.FloatTensor(x))

    def _build_actor(self):
        if self.ouprocess:
            noisy = False
        else:
            noisy = True
        self.actor = Actor(self.n_states, self.n_actions, noisy=noisy,transformer=self.transformer)
        self.actor_criterion = nn.MSELoss()
        self.actor_optimizer = optimizer.Adam(lr=self.alr, params=self.actor.parameters())

    def _build_network(self):
        if self.ouprocess:
            noisy = False
        else:
            noisy = True
        self.actor = Actor(self.n_states, self.n_actions, noisy=noisy,transformer=self.transformer)
        self.target_actor = Actor(self.n_states, self.n_actions,transformer=self.transformer)
        self.critic = Critic(self.n_states, self.n_actions,transformer=self.transformer)
        self.target_critic = Critic(self.n_states, self.n_actions,transformer=self.transformer)

        # if model params are provided, load them
        if len(self.model_name):
            self.load_model(model_name=self.model_name)
            self.logger.info("Loading model from file: {}".format(self.model_name))

        # Copy actor's parameters
        DDPG._update_target(self.target_actor, self.actor, tau=1.0)

        # Copy critic's parameters
        DDPG._update_target(self.target_critic, self.critic, tau=1.0)

        self.loss_criterion = nn.MSELoss()
        self.actor_optimizer = optimizer.Adam(lr=self.alr, params=self.actor.parameters(), weight_decay=1e-5)
        self.critic_optimizer = optimizer.Adam(lr=self.clr, params=self.critic.parameters(), weight_decay=1e-5)

    @staticmethod
    def _update_target(target, source, tau):
        for (target_param, param) in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1 - tau) + param.data * tau
            )

    def reset(self, sigma):
        self.noise.reset(sigma)

    def _sample_batch(self):
        batch, idx = self.replay_memory.sample(self.batch_size)
        # batch = self.replay_memory.sample(self.batch_size)
        states = list(map(lambda x: x[0], batch))
        actions = list(map(lambda x: x[1].tolist(), batch))
        rewards = list(map(lambda x: x[2], batch))
        next_states = list(map(lambda x: x[3], batch))
        terminates = list(map(lambda x: x[4], batch))

        return idx, states, next_states, actions, rewards, terminates

    def add_sample(self, state, action, reward, next_state, terminate):
        self.critic.eval()
        self.actor.eval()
        self.target_critic.eval()
        self.target_actor.eval()
        batch_state = self.normalizer([state])
        batch_next_state = self.normalizer([next_state])
        current_value = self.critic(batch_state, DDPG.totensor([action.tolist()]))
        target_action = self.target_actor(batch_next_state)
        target_value = DDPG.totensor([reward]) \
            + DDPG.totensor([0 if x else 1 for x in [terminate]]) \
            * self.target_critic(batch_next_state, target_action) * self.gamma
        error = float(torch.abs(current_value - target_value).data.numpy()[0])

        self.target_actor.train()
        self.actor.train()
        self.critic.train()
        self.target_critic.train()
        self.replay_memory.add(error, (state, action, reward, next_state, terminate))
        # self.logger.info('replay_memory add state: {}'.format(state))

    def delta_action_value(self, action0, action1, current_value0, current_value1):
        delta_action = abs(action1 - action0).sum(axis=0)
        delta_action=delta_action / delta_action.shape[0]
        delta_value=((current_value1 - current_value0) * (current_value1 - current_value0)).mean()
        self.logger.info('[ddpg] delta_action: {}'.format(delta_action))
        self.logger.info('[ddpg] critic_MSE: {}'.format(delta_value))
        return delta_action,delta_value

    def update(self):
        """ Update the Actor and Critic with a batch data
        """
        idxs, states, next_states, actions, rewards, terminates = self._sample_batch()
        batch_states = self.normalizer(states)
        batch_next_states = self.normalizer(next_states)
        batch_actions = DDPG.totensor(actions)
        batch_rewards = DDPG.totensor(rewards)

        mask = [0 if x else 1 for x in terminates]
        mask = DDPG.totensor(mask)
        if self.debug:
            action0= self.target_actor(batch_states).detach().data.numpy()
            current_value0 = self.target_critic(batch_states, batch_actions).data.numpy()

        target_next_actions = self.target_actor(batch_next_states).detach()
        target_next_value = self.target_critic(batch_next_states, target_next_actions).detach().squeeze(1)

        current_value = self.critic(batch_states, batch_actions)
        next_value = batch_rewards + mask * target_next_value * self.gamma

        # update prioritized memory
        error = torch.abs(current_value - next_value).data.numpy()
        for i in range(self.batch_size):
            idx = idxs[i]
            self.replay_memory.update(idx, error[i][0])

        # Update Critic
        loss = self.loss_criterion(current_value.squeeze(1), next_value)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        self.critic.eval()
        policy_loss = -self.critic(batch_states, self.actor(batch_states))
        policy_loss = policy_loss.mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()

        self.actor_optimizer.step()
        self.critic.train()

        DDPG._update_target(self.target_critic, self.critic, tau=self.tau)
        DDPG._update_target(self.target_actor, self.actor, tau=self.tau)

        if self.debug:    
            action1 = self.target_actor(batch_states).detach().data.numpy()
            current_value1 = self.target_critic(batch_states, batch_actions).data.numpy()
            self.delta_action_value(action0,action1,current_value0,current_value1)

        return loss.item(), policy_loss.item()

    def choose_action(self, x, coff=1):
        """ Select Action according to the current state
        Args:
            x: np.array, current state
        """
        self.actor.eval()
        act = self.actor(self.normalizer([x])).squeeze(0)
        self.actor.train()
        action = act.data.numpy()
        # self.logger.info('Action before OUProcess: {}'.format(action))
        if self.ouprocess:
            action += self.noise.noise()  * coff
        return action.clip(0, 1)

    def sample_noise(self):
        self.actor.sample_noise()

    def load_model(self, model_name):
        """ Load Torch Model from files
        Args:
            model_name: str, model path
        """
        self.actor.load_state_dict(
            torch.load('{}_actor.pth'.format(model_name))
        )
        self.critic.load_state_dict(
            torch.load('{}_critic.pth'.format(model_name))
        )

    def save_model(self, model_dir, title):
        """ Save Torch Model from files
        Args:
            model_dir: str, model dir
            title: str, model name
        """
        torch.save(
            self.actor.state_dict(),
            '{}/{}_actor.pth'.format(model_dir, title)
        )

        torch.save(
            self.critic.state_dict(),
            '{}/{}_critic.pth'.format(model_dir, title)
        )

    def save_actor(self, path):
        """ save actor network
        Args:
             path, str, path to save
        """
        torch.save(
            self.actor.state_dict(),
            path
        )

    def load_actor(self, path):
        """ load actor network
        Args:
             path, str, path to load
        """
        self.actor.load_state_dict(
            torch.load(path)
        )

    def train_actor(self, batch_data, is_train=True):
        """ Train the actor separately with data
        Args:
            batch_data: tuple, (states, actions)
            is_train: bool
        Return:
            _loss: float, training loss
        """
        states, action = batch_data

        if is_train:
            self.actor.train()
            pred = self.actor(self.normalizer(states))
            action = DDPG.totensor(action)

            _loss = self.actor_criterion(pred, action)

            self.actor_optimizer.zero_grad()
            _loss.backward()
            self.actor_optimizer.step()

        else:
            self.actor.eval()
            pred = self.actor(self.normalizer(states))
            action = DDPG.totensor(action)
            _loss = self.actor_criterion(pred, action)

        return _loss.item()

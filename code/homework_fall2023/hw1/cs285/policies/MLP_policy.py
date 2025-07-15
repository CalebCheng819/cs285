"""
Defines a pytorch policy as the agent's actor

Functions to edit:
    2. forward
    3. update
"""

import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            MLP (nn.Module)
    """
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(nn.Tanh())
        in_size = size
    layers.append(nn.Linear(in_size, output_size))

    mlp = nn.Sequential(*layers)
    return mlp


class MLPPolicySL(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    """
    Defines an MLP for supervised learning which maps observations to continuous
    actions.

    Attributes
    ----------
    mean_net: nn.Sequential
        A neural network that outputs the mean for continuous actions
    logstd: nn.Parameter
        A separate parameter to learn the standard deviation of actions

    Methods
    -------
    forward:
        Runs a differentiable forwards pass through the network
    update:
        Trains the policy with a supervised learning objective
    """
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        self.mean_net = build_mlp(#mean_net是一个神经网络，用于输出连续动作的均值,输入和输出的维度分别为ob_dim和ac_dim                                                      
            input_size=self.ob_dim,
            output_size=self.ac_dim,
            n_layers=self.n_layers, size=self.size,
        )#同时学习mean和logstd
        self.mean_net.to(ptu.device)
        self.logstd = nn.Parameter(

            torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)#将logstd作为可学习参数
        )
        self.logstd.to(ptu.device)
        self.optimizer = optim.Adam(
            itertools.chain([self.logstd], self.mean_net.parameters()),
            self.learning_rate
        )

    def save(self, filepath):
        """
        :param filepath: path to save MLP
        """
        torch.save(self.state_dict(), filepath)

    def forward(self, observation: torch.FloatTensor) -> Any:#这个是内部调用函数，故为tensor类型
        """
        Defines the forward pass of the network

        :param observation: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy
        """
        # TODO: implement the forward pass of the network.
        # You can return anything you want, but you should be able to differentiate
        # through it. For example, you can return a torch.FloatTensor. You can also
        # return more flexible objects, such as a
        # `torch.distributions.Distribution` object. It's up to you!
        observation = observation.to(ptu.device)
        mean = self.mean_net(observation)#调用mean_net进行前向传播
        logstd = self.logstd.expand_as(mean)  # logstd is a single value
        std = torch.exp(logstd)  # convert logstd to std
        dist = distributions.Normal(mean, std)  # create a normal distribution  
        action = dist.rsample()  # sample an action from the distribution,使用rsample可以使得梯度可以通过采样传递
        action = torch.tanh(action)  # apply tanh to the action限制
        return action, dist
    
    @torch.no_grad()
    def get_action(self, observation:np.ndarray) -> np.ndarray:
        """
        Returns an action given an observation

        :param observation: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy
        """
        observation = ptu.from_numpy(observation).to(ptu.device)
        action, _ = self.forward(observation)
        return ptu.to_numpy(action)
        

    def update(self, observations, actions):
        """
        Updates/trains the policy

        :param observations: observation(s) to query the policy
        :param actions: actions we want the policy to imitate
        :return:
            dict: 'Training Loss': supervised learning loss
        """
        # TODO: update the policy and return the loss
        observations = ptu.from_numpy(observations).to(ptu.device)#将observations转换为pytorch张量并移动到ptu.device上
        actions = ptu.from_numpy(actions).to(ptu.device)
        self.optimizer.zero_grad()
        _, dist = self.forward(observations)
        # Calculate the loss as the negative log likelihood of the actions under the policy 
        log_probs = dist.log_prob(actions)
        log_probs = log_probs.sum(dim=-1)  # sum over action dimensions
        loss = -log_probs.mean()  # mean over batch
        # Backpropagate the loss
        loss.backward()
        self.optimizer.step()
        # Return a dictionary with the training loss
        # You can add extra logging information here, but keep this line
        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }

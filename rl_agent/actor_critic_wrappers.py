"""
    actor_critic_wrappers.py

    Wrapper classes for actor and critic models for DDPG-RL-like debiaser.
    Adds insulation between model cores and the domain label-handling.
"""

import torch


class RLActor(torch.nn.Module):
    """
        Actor: learns to approximate an observation -> action function.
        'observations' are labelled data items; 'actions' are the same items with
        the same labels but perturbed features. The core, therefore, is a residual
        network whose output the wrapper adds to the features.

        TODO out of date -- do we even still need this file...?
    """

    def __init__(self, dataset, core, initial_factor = 1.0):
        super().__init__()
        
        self.dataset = dataset
        self.core = core
        #self.factor = torch.nn.Parameter(torch.tensor([initial_factor]))
        #self.clamper = torch.nn.Sigmoid()
        #self.num_to_clamp = dataset.n_y + dataset.n_z

    def forward(self, observations):

        #pre_clamp = self.core(observations)

        #return torch.cat((pre_clamp[:, -self.num_to_clamp:], self.clamper(pre_clamp[:, :-self.num_to_clamp])), 1)

        return self.core(observations)



class RLCritic(torch.nn.Module):
    """
        Critic: learns to approximate an (observation, action) -> reward function.
    """
    # TODO: whether the critic gets the labels is an open question.

    def __init__(self, dataset, core):
        super().__init__()

        self.dataset = dataset
        self.core = core

    def forward(self, observations, actions):
        return self.core(torch.concat((observations, actions), axis=1))

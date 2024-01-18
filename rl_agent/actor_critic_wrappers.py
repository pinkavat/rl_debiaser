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
    """

    def __init__(self, dataset, core, initial_factor = 1.0):
        super().__init__()
        
        self.dataset = dataset
        self.core = core
        self.factor = torch.nn.Parameter(torch.tensor([initial_factor]))
        self.clamper = torch.nn.Sigmoid()
        self.num_to_clamp = dataset.n_y + dataset.n_z

    def forward(self, observations):

        pre_clamp = self.core(observations)

        return torch.cat((pre_clamp[:, -self.num_to_clamp:], self.clamper(pre_clamp[:, :-self.num_to_clamp])), 1)

        """
        # Separate observation features and observation labels
        X, y, z, q = self.dataset.split_labels(observations)

        # Pass labels and features into the core; core returns perturbed features
        residue = self.core(observations)

        # Scale the residue by some factor
        residue = residue * self.factor

        # Concatenate original labels with sum of residue and original features
        return self.dataset.attach_labels(residue + X, y, z, q)
    	"""

# TODO: allgen -- prototype and MAKE MODULAR!
"""
    def __init__(self, dataset, core, initial_factor = 1.0):
        super().__init__()
        
        self.dataset = dataset
        self.core = core
        self.factor = torch.nn.Parameter(torch.tensor([initial_factor]))
        self.s = torch.nn.Sigmoid()

    def forward(self, observations):

        # Separate observation features and observation labels
        X, y, z, q = self.dataset.split_labels(observations)

        # Pass labels and features into the core; core returns perturbed features
        residue = self.core(q) # TODO: Nobbut fairness experiment

        # Scale the residue by some factor
        #residue = residue * self.factor

        # Concatenate original labels with sum of residue and original features
        #return self.dataset.attach_labels(residue, y, z, q) # TODO Noresid

        # TODO TODO: exp. allcreate -- HARDCODED, NOT MODULAR
        num_clamped = 3
        return torch.cat((residue[:, -num_clamped:], self.s(residue[:, :-num_clamped])), 1)

"""


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

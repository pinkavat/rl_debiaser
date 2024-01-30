"""
    ddpg_agent.py

    Direct Deterministic Policy Gradient Reinforcement Learning Agent
    with total myopia (no discounted future reward) due to paradigm.
"""

import torch
import copy
import random # TODO better dist

import rl_agent.replay_buffer # TODO bad dep


class Agent():
    """
        DDPG RL Agent: maintains an inner actor/critic Q-learner. Polls the actor to respond to observations of the environment.
        When the environment reacts, stores the reaction in a replay-buffer memory. In the learning step, samples observations
        from the buffer and trains thereon.
    """
    

    def __init__(
        self, dataset=None, actor=None, critic=None, actor_optimizer=None, critic_optimizer=None,
        gamma = 0.99, tau = 0.001, exploration_linear_decay = 0.0004, sample_size=32, memory_size = None, device_override = None
    ):
        """
            Initialize a new Agent, with the given RLActor and RLCritic (actor_critic_wrappers.py) as actor and critic submodels.
            actor_optimizer and critic_optimizers are torch optimization functions.
            sample_size and optional memory_size are hyperparameters.
    	"""
        
        # Detect torch device
        self.device = device_override if device_override else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        self.dataset = dataset
        self.clamper = torch.nn.Sigmoid()
        self.num_to_clamp = dataset.n_y + dataset.n_z

        # Set up Actor and Critic
        self.actor = actor.to(self.device)
        self.actor_optimizer = actor_optimizer
        self.critic = critic.to(self.device)
        self.critic_optimizer = critic_optimizer

        # Set up Actor and Critic Delayed Models (secondary models, used for evaluation on future state only, to whom Actor and Critic weights are softcopied)
        # (these are called 'targets' in traditional DL but we don't want to confuse them with the target model on whom the task is working)
        self.delayed_actor = copy.deepcopy(actor).to(self.device)
        self.delayed_critic = copy.deepcopy(critic).to(self.device)
        self.delayed_actor.eval()
        self.delayed_critic.eval()

        self.gamma = gamma
        self.tau = tau

        self.explore_factor = 1.0
        self.exploration_decay = exploration_linear_decay

        # Set up memory
        self.sample_size = sample_size
        backing_size = sample_size if memory_size is None else memory_size
        self.memory = rl_agent.replay_buffer.ReplayBuffer(backing_size if backing_size > sample_size else sample_size)



    def episode_reset(self):
        """ Resets the agent's state; to be invoked at the beginning of every episode. """
        self.memory.clear()
        self.explore_factor = 1.0



    def act_on(self, observation_batch, exploration = True):
        """ Decides on an action given the given observation. """

        with torch.no_grad():
            self.actor.eval()
            action_batch = self.actor(observation_batch.to(self.device))

        # Apply exploration perturbation
        for action in action_batch:

            # TODO better process, distrib params, etc.
            perturbation = [random.uniform(-100.0, 100.0) for x in range(action.size(0))]

            action += float(exploration) * self.explore_factor * torch.tensor(perturbation)

        # Clamp labels
        action_batch = torch.cat((action_batch[:, :-self.num_to_clamp], self.clamper(action_batch[:, -self.num_to_clamp:])), 1) 

        # Decay exploration factor
        self.explore_factor = max(0.0, self.explore_factor - self.exploration_decay)

        return action_batch



    def estimate_future_reward(self, next_state_batch):
        """ Returns the delayed critic prediction for the delayed actor decision for the given next-state batch. Diagnostic tool. """
        with torch.no_grad():
            return self.delayed_critic(torch.cat((next_state_batch, self.delayed_actor(next_state_batch)), 1))
    


    def observe(self, state, action, reward, next_state):
        """ Observe how the environment reacts to our given action for the given observation; store into memory. """
        
        self.memory.store((state, action, reward, next_state))



    def learn_from_memory(self, train_actor = True):
        """ The heart of the DDPG RL technique: sample N observations from memory and train the actor/critic Q learner thereon. """
        
        if len(self.memory) >= self.sample_size: # TODO how to react to being in 'warmup'?
           
            # Sample from memory
            states, actions, rewards, next_states = self.memory.sample_into_batches(self.sample_size)

            # Flatten batches and dispatch to device
            states_batch = torch.cat(states).to(self.device)
            actions_batch = torch.cat(actions).to(self.device)
            rewards_batch = torch.cat(rewards).to(self.device)
            next_states_batch = torch.cat(next_states).to(self.device)

            # Delayed critic estimates future reward
            with torch.no_grad():
                delayed_q_pred = rewards_batch + self.gamma * self.delayed_critic(torch.cat((next_states_batch, self.delayed_actor(next_states_batch)), 1))

            # Critic training pass
            self.critic.train()
            self.critic_optimizer.zero_grad()

            q_pred = self.critic(torch.cat((states_batch, actions_batch), 1))
            critic_loss = self.dataset.critic_loss(q_pred, delayed_q_pred)
        
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor training pass
            if(train_actor):
                self.actor.train()
                self.critic.eval()
                self.actor_optimizer.zero_grad()

                actions_pred = self.actor(states_batch)
                actor_loss = -self.critic(torch.cat((states_batch, actions_pred), 1)).mean()
                actor_loss.backward()
                self.actor_optimizer.step()

            # Soft-copy weights
            self.soft_copy_weights(self.actor, self.delayed_actor)
            self.soft_copy_weights(self.critic, self.delayed_critic)


    def save_to(self, path):
        """ 
            Serializes and saves actor and critic model parameters and optimizer parameters 
            to the given path. Doesn't save model architecture or the replay buffer contents;
            used for checkpointing.
        """
        try:
            torch.save({
                'actor_model_params' : self.actor.state_dict(),
                'actor_optim_params' : self.actor_optimizer.state_dict(),
                'critic_model_params' : self.critic.state_dict(),
                'critic_optim_params' : self.critic_optimizer.state_dict(),
                }, path)
            print(f"Agent parameters saved to {path}")
        except:
            print(f"Couldn't save agent parameters to {path}")


    def load_from(self, path):
        """ 
            Loads actor and critic model and optimizer parameters from the given path;
            the inverse of save_to (above), with the same limitations.
        """
        try:
            state = torch.load(path)
            self.actor.load_state_dict(state['actor_model_params'])
            self.actor_optimizer.load_state_dict(state['actor_optim_params'])
            self.critic.load_state_dict(state['critic_model_params'])
            self.critic_optimizer.load_state_dict(state['critic_optim_params'])
            print(f"Agent parameters loaded from {path}")
        except:
            print(f"Couldn't load agent parameters from {path}")


    def soft_copy_weights(self, source, target):
        """
            Softcopy a model's weights to a delayed model, by the copy factor
            Code from https://github.com/ghliu/pytorch-ddpg/blob/master/util.py as it's very simple
        """
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + source_param.data * self.tau)

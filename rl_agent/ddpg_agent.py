"""
    ddpg_agent.py

    Direct Deterministic Policy Gradient Reinforcement Learning Agent
    with total myopia (no discounted future reward) due to paradigm.
"""

import torch

import rl_agent.replay_buffer # TODO bad dep


class Agent():
    """
        DDPG RL Agent: maintains an inner actor/critic Q-learner. Polls the actor to respond to observations of the environment.
        When the environment reacts, stores the reaction in a replay-buffer memory. In the learning step, samples observations
        from the buffer and trains thereon.
    """
    

    def __init__(self, dataset=None, actor=None, critic=None, actor_optimizer=None, critic_optimizer=None, sample_size=32, memory_size = None, device_override = None):
        """
            Initialize a new Agent, with the given RLActor and RLCritic (actor_critic_wrappers.py) as actor and critic submodels.
            actor_optimizer and critic_optimizers are torch optimization functions.
            sample_size and optional memory_size are hyperparameters.
    	"""
        
        # Detect torch device
        self.device = device_override if device_override else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        self.dataset = dataset
        self.actor = actor.to(self.device)
        self.actor_optimizer = actor_optimizer
        self.critic = critic.to(self.device)
        self.critic_optimizer = critic_optimizer

        # Set up memory
        self.sample_size = sample_size
        backing_size = sample_size if memory_size is None else memory_size
        self.memory = rl_agent.replay_buffer.ReplayBuffer(backing_size if backing_size > sample_size else sample_size)



    def episode_reset(self):
        """ Resets the agent's state; to be invoked at the beginning of every episode. """

        self.memory.clear()



    def act_on(self, observation):
        """ Decides on an action given the given observation. """

        with torch.no_grad():
            self.actor.eval()
            return self.actor(observation.to(self.device))

    

    def observe_action_reward(self, observation, action, reward):
        """ Observe how the environment reacts to our given action for the given observation; store into memory. """
        
        self.memory.store((observation, action, reward))



    def learn_from_memory(self, train_actor = True):
        """ The heart of the DDPG RL technique: sample N observations from memory and train the actor/critic Q learner thereon. """
        
        if len(self.memory) >= self.sample_size: # TODO how to react to being in 'warmup'?
           
            # Sample from memory
            observations, actions, rewards = self.memory.sample_into_batches(self.sample_size)

            # Flatten batches and dispatch to device
            observations_batch = torch.cat(observations).to(self.device)
            actions_batch = torch.cat(actions).to(self.device)
            rewards_batch = torch.cat(rewards).to(self.device)

            # Critic training pass
            self.critic.train()
            self.critic_optimizer.zero_grad()

            q_pred = self.critic(observations_batch, actions_batch)
            critic_loss = self.dataset.critic_loss(q_pred, rewards_batch)
        
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor training pass
            if(train_actor):
                self.actor.train()
                self.critic.eval()
                self.actor_optimizer.zero_grad()

                actions_pred = self.actor(observations_batch)
                actor_loss = -self.critic(observations_batch, actions_pred).mean()
                actor_loss.backward()
                self.actor_optimizer.step()


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

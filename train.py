"""
    train.py

    Takes a hyperparameter/architecture spec and runs the episode loop.
"""

import torch
import time

from rl_agent.actor_critic_wrappers import RLActor, RLCritic
from rl_agent.ddpg_agent import Agent

DEFAULT_EPISODES = 10

DEFAULT_AGENT_MEMORY = 16
DEFAULT_AGENT_SAMPLE = 16

TODO_STATE_SIZE = 2 # TODO Number of items in the 'state'; the input passed to the actor. A TASK SPECIFICATION, rather than a hardcoded thing, ideally.

def run(dataset, target, spec, log_dir = 'logs/'):
    """
        TODO DOCUMENT

        Temp hyperparameters:
            name -- descriptive
            episodes -- default 10
            actor_learn_start -- default 0 (learn at once)
            target_reset -- default 1 (every ep)
            critic_reset -- default nonexistent (if existent, episode reset)
            repetitions -- default 1 (how many times a data item is flogged thru target)

            agent_memory_size
            agent_sample_size

            
            actor_optimizer
            actor_optimizer_params:
                {lr default 1e-3, etc.}
            critic_optimizer
            critic_optimizer_params:
                {lr default 1e-3, etc.}

            actor_core_spec: dictionary. Either:
                {"hidden_layers":[...]} or {"hidden_count":n, "hidden_size":n}

            ditto critic_core_spec.


        TODO hyperparameters:
            Core num layers
            Separate cores Act/critic
    """

    run_name = spec.get('name', f"unnamed_{int(time.time() * 1000.0)}")
    log_path = log_dir + run_name + ".csv"
    log(log_path, ['episode', run_name + "_accuracy", run_name + "_fairness", run_name+"_EO", run_name + "_mean_reward", run_name + "_min_reward", run_name + "_max_reward"], mode='w')

    print(f"Starting run {run_name}")
    print("{")
    for k, v in spec.items():
        print(f"    {k}: {v},")
    print("}\n")


    # Set up actor and critic models
    class Perceptron(torch.nn.Module):
        def __init__(self, layer_spec):
            super().__init__()

            layers = [f(layer_spec[x], layer_spec[x+1]) # note to self: cure addiction to list comprehensions 
                for x in range(len(layer_spec) - 1) 
                for f in (lambda m,n: torch.nn.Linear(m,n), lambda m,n:torch.nn.PReLU())]
            layers 
            self.perceptron = torch.nn.Sequential(*layers[:-1]) # Trim the last ReLU

        def forward(self, X):
            return self.perceptron(X)


    actor_layer_spec = get_perceptron_spec(spec.get('actor_core_spec', {}), TODO_STATE_SIZE, dataset.data_item_size)
    actor = RLActor(dataset, Perceptron(actor_layer_spec))

    critic_layer_spec = get_perceptron_spec(spec.get('critic_core_spec', {}), TODO_STATE_SIZE + dataset.data_item_size, 1) # TODO: past state add here?
    critic = RLCritic(dataset, Perceptron(critic_layer_spec))

    actor_optimizer_fn = spec.get('actor_optimizer', torch.optim.Adam)
    actor_optimizer = actor_optimizer_fn(actor.parameters(), **spec.get('actor_optimizer_params', {'lr':1e-3}))
    critic_optimizer_fn = spec.get('critic_optimizer', torch.optim.Adam)
    critic_optimizer = critic_optimizer_fn(critic.parameters(), **spec.get('critic_optimizer_params', {'lr':1e-3}))


    # Set up agent
    agent = Agent(dataset, actor, critic, actor_optimizer, critic_optimizer, sample_size=spec.get('agent_sample_size', DEFAULT_AGENT_SAMPLE), memory_size=spec.get('agent_memory_size', DEFAULT_AGENT_MEMORY), device_override='cpu')
    print(f"DDPG Agent loaded on {agent.device} device")


    # Report initial state
    target.reset()
    print(f"\nInitial target accuracy: {target.get_accuracy()}\nInitial target independence: {target.get_independence()}\nInitial target EO violation: {target.get_max_equalized_odds_violation()}")

    # Log initial state
    log(log_path, [0, target.get_accuracy(), target.get_independence(),target.get_max_equalized_odds_violation(), 0.0, 0.0, 0.0]) # TODO initial reward meaningful val?


    
    # Enter training loop
    for episode in range(spec.get('episodes', DEFAULT_EPISODES)): # "episode" for RL, not "epoch"
        
        print(f"Episode {episode}:")

        agent.episode_reset()
        if(episode % spec.get('target_reset', 1) == 0):
            target.reset()

        # Critic Resetting, if specified
        if 'critic_reset' in spec:
            if episode % spec['critic_reset'] == 0:
                for layer in agent.critic.children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()

        # ========== Train ==========

        current_obs, total_obs = 0, len(dataset.get_training_data())
        rewards = []

        # Initial state setup
        prev_metric = target.get_max_equalized_odds_violation()
        state = torch.cat((torch.full((dataset.batch_size, 1), prev_metric), torch.full((dataset.batch_size, 1), 0.0) ), axis=1)

        for datum in dataset.get_training_data(): # TODO: No longer perturbing -- passes, instead

            action = agent.act_on(state)

            reward = temp_react(action, target, spec.get('repetitions', 1)) # TODO this isn't actually multiflog -- need to ASSEMBLE a dataset, or maybe jump batch size?
            rewards = rewards + [r.item() for r in reward]

            # Measure current target state, to be recorded as 'future state' by the agent
            metric = target.get_max_equalized_odds_violation()
            delta_metric = metric - prev_metric
            prev_metric = metric
            next_state = torch.cat((torch.full((dataset.batch_size, 1), metric), torch.full((dataset.batch_size, 1), delta_metric) ), axis=1)
           
            agent.observe_action_reward(state, action, reward, next_state)
    
            # Now the future state becomes the present_state for the next pass
            state = next_state

            agent.learn_from_memory(train_actor = episode >= spec.get('actor_learn_start', 0))

            print_progress_bar(current_obs, total_obs)
            print(f" {current_obs} / {total_obs} ", end='')
            current_obs += 1
        print('')

        # ========== Test ===========

        # TODO true test step!
        print(f"\ttrain accuracy: {target.get_accuracy()}\n\ttrain independence: {target.get_independence()}\n\ttrain EO violation: {target.get_max_equalized_odds_violation()}\n")

        # Log to file
        mean_reward = float(sum(rewards)) / float(len(rewards))
        log(log_path, [episode + 1, target.get_accuracy(), target.get_independence(), target.get_max_equalized_odds_violation(), mean_reward, min(rewards), max(rewards)])


        # Checkpoint-save if needful
        if (episode % 10 == 0):
            agent.save_to(f'temp/checkpoints/episode_{episode}.tar')




# Temporary Environment reactor
def temp_react(action_batch, target, repetitions, use_eo = True): # TODO SPEC IT

    # TODO: new: optimizing for minimizing EO    
    previous_reward = 0.0 - target.get_max_equalized_odds_violation() if use_eo else target.get_independence()
    results = []

    # Disintegrate batched action into its constituents
    for action in action_batch:

        # Reshape action into a batch of one
        for repetition in range(repetitions):
            target.train([action.unsqueeze(0)]) # Note the list-clash here, resolve in env. 

        current_reward = 0.0 - target.get_max_equalized_odds_violation() if use_eo else target.get_independence()
        
        reward = current_reward - previous_reward

        previous_reward = current_reward

        results.append(torch.tensor([reward]))
        

    return torch.stack(results)


    """ 
    previous_reward = target.get_independence()
    results = []

    # Disintegrate batched action into its constituents
    for action in action_batch:

        # Reshape action into a batch of one
        for repetition in range(repetitions):
            target.train([action.unsqueeze(0)]) # Note the list-clash here, resolve in env. 

        current_independence = target.get_independence()
        reward = current_independence - previous_reward

        previous_reward = current_independence

        results.append(torch.tensor([reward]))

    return torch.stack(results)
    """


# Progress bar displayer (cribbed from old code)
def print_progress_bar(val, total):
    BAR_WIDTH = 40
    w = (val / float(total)) * BAR_WIDTH
    bar_string = ''.join([' ' if x > w else '=' for x in range(BAR_WIDTH)])
    print(f"\r[{bar_string}]",end='')


# CSV Logfile append
def log(path, items, mode='a'):
    with open(path, mode) as fp:
        fp.write(",".join([str(item) for item in items]))
        fp.write("\n")


# Helper for parsing perceptron core specifications
def get_perceptron_spec(spec, in_n, out_n):
    if 'hidden_layers' in spec:
        # Manual layer specification
        return [in_n, *(spec['hidden_layers']), out_n]
    else:
        # Num and counts given
        hidden_count = spec.get('hidden_count', 2)
        hidden_size = spec.get('hidden_size', 100)
        return [in_n, *[hidden_size for c in range(hidden_count)], out_n]

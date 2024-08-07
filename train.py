"""
    train.py

    Takes a hyperparameter/architecture spec and runs the episode loop.
"""

import torch
import time
import random

from rl_agent.ddpg_agent import Agent
from specced_perceptron import SpeccedPerceptron

TODO_STATE_SIZE = 2 # TODO Number of items in the 'state'; the input passed to the actor. A TASK SPECIFICATION, rather than a hardcoded thing, ideally.

LOG_CRITIC_LOSS = False # TODO

SAMPLE_CURRENT_METRIC_INTERVAL = 32



# Hyperparameter defaults: name -> (default value, descriptor, descriptive group)
DEFAULTS = {
    
    'name' : ('unnamed_<time>', "Descriptive unique name for the run.", 'group_meta'),
    'use_eo' : (True, "If true, minimize Equalized Odds violation; if false, maximize Demographic Parity.", 'group_meta'),

    'episodes' : (10, "Number of episodes: each episode is a series of steps, bookended by target reset and agent memory clear.", 'group_steps'),
    'steps' : (32768, "Number of data items generated by the agent per episode.", 'group_steps'),
    'step_schedule' : ([], "Iterable defining variable number of steps per episode; episodes out of its range revert to step count specified in steps.", 'group_steps'),

    'agent_sample_size' : (1024, "Size of batch of obs/act/rew/next tuples taken by the agent at each learn-from-memory call.", 'group_agent_memory'),
    'agent_memory_size' : (4096, "Number of obs/act/rew/next tuples stored in the agent's recall buffer.", 'group_agent_memory'),

    'agent_gamma' : (0.99, "Prediction of future state discount factor.", 'group_agent_future'),
    'agent_tau' : (0.001, "Prediction of future state to-delayed softcopy coefficient.", 'group_agent_future'),

    'agent_explore_mu' : (0.0, "Mean of Gaussian random exploration process.", 'group_agent_explore'),
    'agent_explore_sigma' : (None, "Standard Deviation of Gaussian random exploration process. If specified, engages exploration", 'group_agent_explore'),
    'agent_explore_decay' : (None, "Proportion of step at which exploratory noise will have decayed to zero. If not specified, exploration does not decay."),

    'actor_optimizer' : (torch.optim.Adam, "Optimizer for Agent Actor net.", 'group_agent_optimizer'),
    'actor_optimizer_params' : ({'lr':1e-6}, "Parameters for the Actor Optimizer; passed through.", 'group_agent_optimizer'),
    'critic_optimizer' : (torch.optim.Adam, "Optimizer for Agent Critic net.", 'group_agent_optimizer'),
    'critic_optimizer_params' : ({'lr':1e-5}, "Parameters for the Critic Optimizer; passed through.", 'group_agent_optimizer'),

    'actor_core_spec' : ({'hidden_layers':[400, 300]}, "Perceptron layer spec for Agent Actor. Either: {'hidden_layers':[...]} or {'hidden_count':n, 'hidden_size':n}",'group_agent_core_spec'),
    'critic_core_spec' : ({'hidden_layers':[400, 300]}, "Perceptron layer spec for Critic Actor; ditto 'actor_core_spec'.", 'group_agent_core_spec'),
}





def run(dataset, target, spec, log_dir = 'logs/', agent_device_override='cpu'):
    """
        Execute a trial with the given dataset, target model, and trial specification.
        agent_device_override is passed to the Agent to override device detection.
    """

    run_name = spec.get('name', f"unnamed_{int(time.time() * 1000.0)}")
    log_path = log_dir + run_name + ".csv"
    critic_loss_log_path = log_dir + run_name + "_critic_loss_" + ".csv"
    #log(log_path, ['episode', 'step', run_name + "_accuracy", run_name + "_fairness", run_name+"_EO", run_name + "_mean_reward", run_name + "_min_reward", run_name + "_max_reward", run_name + "_mean_future_pred_rew"], mode='w')
    log(log_path, ['episode', 'substep', 'is_test', run_name + "_accuracy", run_name + "_dem_par", run_name+"_EO", run_name+"_deltaEO"], mode='w')

    if LOG_CRITIC_LOSS:
        log(critic_loss_log_path, ['episode', 'substep', run_name + '_critic_loss'], mode='w')

    print(f"Starting run {run_name}")
    print("{")
    for k, v in spec.items():
        print(f"    {k}: {v},")
    print("}\n")


    # Set up actor and critic models
    actor = SpeccedPerceptron.from_text_spec(spec.get('actor_core_spec', DEFAULTS['actor_core_spec'][0]), TODO_STATE_SIZE, dataset.data_item_size)
    critic = SpeccedPerceptron.from_text_spec(spec.get('critic_core_spec', DEFAULTS['critic_core_spec'][0]), TODO_STATE_SIZE + dataset.data_item_size, 1)

    actor_optimizer_fn = spec.get('actor_optimizer', DEFAULTS['actor_optimizer'][0])
    actor_optimizer = actor_optimizer_fn(actor.parameters(), **spec.get('actor_optimizer_params', DEFAULTS['actor_optimizer_params'][0]))
    critic_optimizer_fn = spec.get('critic_optimizer', DEFAULTS['critic_optimizer'][0])
    critic_optimizer = critic_optimizer_fn(critic.parameters(), **spec.get('critic_optimizer_params', DEFAULTS['critic_optimizer_params'][0]))
    
    exploration_process = (lambda : random.gauss(spec.get('agent_explore_mu', DEFAULTS['agent_explore_mu'][0]), spec['agent_explore_sigma'])) if 'agent_explore_sigma' in spec else None

    # Set up agent
    agent = Agent(dataset, actor, critic, actor_optimizer, critic_optimizer,
        spec.get("agent_gamma", DEFAULTS['agent_gamma'][0]), spec.get("agent_tau", DEFAULTS['agent_tau'][0]),
        exploration_process = exploration_process,
        sample_size=spec.get('agent_sample_size', DEFAULTS['agent_sample_size'][0]), 
        memory_size=spec.get('agent_memory_size', DEFAULTS['agent_memory_size'][0]), 
        device_override=agent_device_override)
    print(f"DDPG Agent loaded on {agent.device} device")


    # Report initial state
    target.reset()
    print(f"\nInitial target accuracy: {target.get_accuracy(validation=True)}")
    print(f"Initial target independence: {target.get_independence(validation=True)}")
    print(f"\x1b[1mInitial target EO violation: {target.get_max_equalized_odds_violation(validation=True)}\x1b[0m")

    # Log initial state
    log(log_path, [0, 0, 1, target.get_accuracy(validation=True), target.get_independence(validation=True), target.get_max_equalized_odds_violation(validation=True), 0.0])
    overall_step = 1
    terminal_eo = target.get_max_equalized_odds_violation(validation=True)
    
    # Enter training loop
    for episode in range(spec.get('episodes', DEFAULTS['episodes'][0])): # "episode" for RL, not "epoch"
        
        print(f"Episode {episode}:")

        # ========== Train ==========

        step_schedule = spec.get('step_schedule', [])
        steps = step_schedule[episode] if episode < len(step_schedule) else spec.get('steps', DEFAULTS['steps'][0])

        # Agent Reset, with decay control
        decay_fraction = spec.get('agent_explore_decay', DEFAULTS['agent_explore_decay'][0])
        if decay_fraction:
            # Enable decay for this step
            decay_per_step = 1.0 / (steps * decay_fraction)
            agent.episode_reset(exploration_decay = decay_per_step)
        else:
            # Disable decay for this step
            agent.episode_reset() # Decay defaults to zero

        # Target reset
        target.reset()

        use_eo = DEFAULTS['use_eo'][0]


        # Initial state setup
        prev_metric = 0.0 - target.get_max_equalized_odds_violation() if use_eo else target.get_independence()
        state = torch.tensor([prev_metric, 0.0])

        for step in range(steps):

            # 1) Agent acts on single state
            action = agent.act_on(state.unsqueeze(0)) # Note that agent expects batch dimension, and returns batched.

            # 2) Environment reacts
            target.train([action]) # TODO repetitions?

            current_metric = 0.0 - target.get_max_equalized_odds_violation() if use_eo else target.get_independence()
            delta_metric = current_metric - prev_metric
            prev_metric = current_metric

            # Log training result
            log(log_path, [episode + 1, overall_step, 0, target.get_accuracy(), target.get_independence(), target.get_max_equalized_odds_violation(), 0.0 - delta_metric])
            overall_step += 1
            
            next_state = torch.tensor([current_metric, delta_metric])
            
            # 3) Store observations in the agent's memory. Note that the agent expects a batch dimension.
            agent.observe(state.unsqueeze(0), action, torch.tensor([delta_metric]).unsqueeze(0), next_state.unsqueeze(0))
            
            # 4) Advance state for the next pass
            state = next_state

            # 5) Agent learns, if it's allowed to
            critic_loss = agent.learn_from_memory()
            
            if LOG_CRITIC_LOSS:
                log(critic_loss_log_path, [episode + 1, step + episode * steps, critic_loss])

            if (step % SAMPLE_CURRENT_METRIC_INTERVAL == 0):
                sampled_train_metric = target.get_max_equalized_odds_violation() if use_eo else target.get_independence()
            

            print_progress_bar(step, steps)
            print(f" {step} / {steps}\tcur train: {sampled_train_metric}\t\t   ", end='')
        print('')



        # ========== Test ===========

        test_eo = target.get_max_equalized_odds_violation(validation=True)

        print(f"\ttest accuracy: {target.get_accuracy(validation=True)}")
        print(f"\ttest independence: {target.get_independence(validation=True)}")
        print(f"\t\x1b[1mtest EO violation: {test_eo}\x1b[0m")
        print(f"\ttrain EO violation: {target.get_max_equalized_odds_violation(validation=False)}")

        log(log_path, [episode + 1, overall_step, 1, target.get_accuracy(validation=True), target.get_independence(validation=True), target.get_max_equalized_odds_violation(validation=True), 0.0])

        terminal_eo = test_eo


        # Checkpoint-save if needful (TODO overhaul; add reentry mechanism)
        if (episode % 5 == 0):
            agent.save_to(f'temp/checkpoints/episode_{episode}.tar')

    return terminal_eo



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

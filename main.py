"""
    main.py

    Driver Code for 'Cleaner RL' October Experiments
"""

import torch
import numpy as np  # For seed-setting
import random # Ditto

import train
import adult_dataset_handler
import compas_dataset_handler
import target_nn_binary


# TODO List
#
#   Technical:
#       - Clean test data, followed by True Test Step
#       - Multiclass fairness
#       - K-fold cross validation
#
#   Theoretical:
#       - Does the critic get the labels?
#       - How is sensitivity test data gathered? From what population and what sampling strategy?
#       - How MANY data items do we need for a sensitivity estimate?
#       - How MANY data items do we need per perturbation?
#       - "Rolling fairness" (see notes)
#       - Algorithm may be prone to fall into local minima
#       - Reward function: direct bias or delta bias; parametrized accuracy term?
#       - Age considerations:
#           - When to reset the target to its initial parameters
#           - Model is inferring as target is evolving
#           - Age parameters? Privilege earlier steps? Keep earlier data-items in memory?
#       - Training data for the initial victim: is it OK if it's the same data pool we use to run the debiaser?
#       - Target-style actor copying
#       - Sim annealing optimizer
#       - Cool-off reset: reset more earlier, then less often?
#       - Lower batch size






# Seed PRNG and set for deterministic running
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
random.seed(0)


# Set up dataset and target model

# A) ADULT TASK
# TODO note train size reduced for functionality testing
dataset = adult_dataset_handler.AdultDataset('../adult/adult.data', train_size = 0.2)
target = target_nn_binary.RLTarget(dataset, device_override='cpu', parameter_path='temp/targ_params_adult.pt')

# or B) COMPAS TASK
#dataset = compas_dataset_handler.COMPASDataset('../compas/compas_scores_two_years_clean.csv')
#target = target_nn_binary.RLTarget(dataset, device_override='cpu', parameter_path='temp/targ_params_compas.pt')


print('')


# SUPERSIZED SYSTEM TESTS

def supersized_system_tests(act_count, act_size, crit_count, crit_size, episodes = 10, name=None, alr=1e-3, clr=1e-3):
    train.run(dataset, target, {
        'name' : name if name else f"a{act_count}:{act_size}-c{crit_count}:{crit_size}",
        'actor_core_spec' : {'hidden_count':act_count, 'hidden_size':act_size},
        'critic_core_spec' : {'hidden_count':crit_count, 'hidden_size':crit_size},
        'episodes' : episodes,
        
        'actor_optimizer_params' : {'lr':alr},
        'critic_optimizer_params' : {'lr':clr},
    })


supersized_system_tests(2,300,2,300,episodes=20,alr=1e-4,clr=1e-3,name="a2:300-c2:300_3")

#train.run(dataset, target, {
#    'name' : "test_",
#    'episodes' : 300,
#    'actor_core_spec' : {'hidden_layers':[400,300]},
#    'actor_optimizer_params' : {'lr':1e-4},
#    'critic_core_spec' : {'hidden_layers':[400,300]},
#    'critic_optimizer_params' : {'lr':1e-3},
#})

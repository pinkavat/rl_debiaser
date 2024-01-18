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
import target_nn_binary as target_nn_adult


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






# Seed PRNG
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
random.seed(0)


# Set up dataset and domain interface
dataset = adult_dataset_handler.AdultDataset('../adult/adult.data', train_batch_size = 32, test_batch_size = 32) # TODO batch size hyperparam
#dataset = compas_dataset_handler.COMPASDataset('../compas/compas_scores_two_years_clean.csv', train_batch_size = 32, test_batch_size = 32) # TODO batch size hyperparam

# Set up the target model
target = target_nn_adult.RLTarget(dataset, dataset.get_target_eval_data(), training_data = dataset.get_training_data(), device_override='cpu', parameter_path='temp/params_target.pt') # TODO training/testing data source
# TODO TODO TODO: sample size problems etc. Also we're reaching into the dataset -- wrap in method later.
# TODO: bad dep again. Shouldn't be the responsibility of the main to worry about how the target tests its fairness.
print('')


# SUPERSIZED SYSTEM TESTS

def supersized_system_tests(act_count, act_size, crit_count, crit_size, episodes = 10, name=None):
    train.run(dataset, target, {
        'name' : name if name else f"a{act_count}:{act_size}-c{crit_count}:{crit_size}",
        'actor_core_spec' : {'hidden_count':act_count, 'hidden_size':act_size},
        'critic_core_spec' : {'hidden_count':crit_count, 'hidden_size':crit_size},
        'episodes' : episodes,
    })

# TODO autoenumeration of repeat tests?

train.run(dataset, target, {
    'name' : "test_IGNORE",
    'episodes' : 10,
})

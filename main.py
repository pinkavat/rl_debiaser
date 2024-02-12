"""
    main.py

    Driver Code for 'Cleaner RL' October Experiments
"""

import torch
import numpy as np  # For seed-setting
import random # Ditto
import copy

import train
import adult_dataset_handler
import compas_dataset_handler
import german_dataset_handler
import target_nn_binary





# Seed PRNG and set for deterministic running
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
random.seed(0)


# Set up dataset and target model

# A) ADULT TASK
# TODO note train size reduced for functionality testing
dataset = adult_dataset_handler.AdultDataset('../adult/adult.data', train_size = 0.3, test_size = 0.5)
target = target_nn_binary.RLTarget(dataset, device_override='cpu', parameter_path='temp/targ_params_adult.pt')

# or B) COMPAS TASK
#dataset = compas_dataset_handler.COMPASDataset('../compas/compas_scores_two_years_clean.csv')
#target = target_nn_binary.RLTarget(dataset, device_override='cpu', parameter_path='temp/targ_params_compas.pt')

# or C) GERMAN CREDIT TASK
#dataset = german_dataset_handler.GermanDataset('../german/german.data')
#target = target_nn_binary.RLTarget(dataset, device_override='cpu', parameter_path='temp/targ_params_german.pt')

print('')




train.run(dataset, target, {
    'name': "sig_8.0_no_learning_40000s",
    'episodes' : 3,
    'steps' : 40000,

    'agent_explore_sigma' : 8.0,
    'actor_optimizer_params' : {'lr' : 0.0},
    'critic_optimizer_params' : {'lr' : 0.0},
})

train.run(dataset, target, {
    'name': "sig_8.0_B_40000s",
    'episodes' : 10,
    'steps' : 40000,

    'agent_explore_sigma' : 8.0,
    'actor_optimizer_params' : {'lr' : 1e-7},
    'critic_optimizer_params' : {'lr' : 1e-6},
})

train.run(dataset, target, {
    'name': "sig_8.0_B_20000s",
    'episodes' : 10,
    'steps' : 20000,

    'agent_explore_sigma' : 8.0,
    'actor_optimizer_params' : {'lr' : 1e-7},
    'critic_optimizer_params' : {'lr' : 1e-6},
})

train.run(dataset, target, {
    'name': "sig_8.0_B_60000s",
    'episodes' : 10,
    'steps' : 60000,

    'agent_explore_sigma' : 8.0,
    'actor_optimizer_params' : {'lr' : 1e-7},
    'critic_optimizer_params' : {'lr' : 1e-6},
})

train.run(dataset, target, {
    'name': "sig_8.0_B_80000s",
    'episodes' : 5,
    'steps' : 80000,

    'agent_explore_sigma' : 8.0,
    'actor_optimizer_params' : {'lr' : 1e-7},
    'critic_optimizer_params' : {'lr' : 1e-6},
})

train.run(dataset, target, {
    'name': "sig_8.0_no_learning_20000s",
    'episodes' : 3,
    'steps' : 20000,

    'agent_explore_sigma' : 8.0,
    'actor_optimizer_params' : {'lr' : 0.0},
    'critic_optimizer_params' : {'lr' : 0.0},
})

train.run(dataset, target, {
    'name': "sig_8.0_no_learning_60000s",
    'episodes' : 3,
    'steps' : 60000,

    'agent_explore_sigma' : 8.0,
    'actor_optimizer_params' : {'lr' : 0.0},
    'critic_optimizer_params' : {'lr' : 0.0},
})

train.run(dataset, target, {
    'name': "sig_8.0_no_learning_80000s",
    'episodes' : 3,
    'steps' : 80000,

    'agent_explore_sigma' : 8.0,
    'actor_optimizer_params' : {'lr' : 0.0},
    'critic_optimizer_params' : {'lr' : 0.0},
})




"""
REPETITIONS_OF_PROMISING = 3
PROMISING_THRESHOLD = 0.09
specs_to_try = [



    {
        'name' : "1em6_1em5",

        'episodes' : 2,
        #'steps' : 32768, # TODO mul 32
        #'step_schedule' : [16, 16, 32, 32, 64, 64, 32, 32, 16],

        'agent_explore_sigma' : 8.0,

        #'actor_optimizer_params': {'lr': 2e-6},
        #'critic_optimizer_params': {'lr': 2e-5},

        #'agent_sample_size' : 1024, # TODO s32m128 muls by 32 now
        #'agent_memory_size' : 4096,
    },



]


for source_spec in specs_to_try:

    for repetition in range(REPETITIONS_OF_PROMISING):
        spec = copy.deepcopy(source_spec)
        spec['name'] = f"{spec.get('name', 'unnamed')}_{repetition + 1}"
        spec['promising_threshold'] = PROMISING_THRESHOLD
        promising = train.run(dataset, target, spec)

        if not promising:
            # Stop trial
            print("\nNot promising -- moving on\n")
            break
        else:
            print("\nPromising -- repeating trial\n")
"""

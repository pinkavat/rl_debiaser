"""
    generate_adversarial_benchmarks.py

    Our implementation of Yang et al. (https://doi.org/10.1038/s41746-023-00805-y)'s adversarial debiaser.
    Used to obtain a secondary benchmark for debiasing performance in the Adult and COMPAS tasks.
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import math
import sys

import torch

# Fetch the datasets, reusing the loaders from the main program
sys.path.append('..')
import adult_dataset_handler
import compas_dataset_handler

adult_dataset = adult_dataset_handler.AdultDataset('../../adult/adult.data')
compas_dataset = compas_dataset_handler.COMPASDataset('../../compas/compas_scores_two_years_clean.csv')

# TASK CONTROL
task_dataset = compas_dataset



# ========== MODEL SPEC ==========

# Yang et al's weight factor: Given accuracy loss Lp on main model and accuracy loss La on adversary, loss given to main model is
# Lp + (Lp / La) - ALPHA * La
ALPHA = 10.0 # TODO: PARAMETRIZE?



class TempModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Model setup here
        self.test_steps = torch.nn.Sequential(
            torch.nn.Linear(task_dataset.n_x, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, X):
        return self.test_steps(X)


class TempAdversary(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Model setup here
        self.test_steps = torch.nn.Sequential(
            torch.nn.Linear(2, 100),         ## y' and y  --> z'
            torch.nn.ReLU(),
            torch.nn.Linear(100, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, X):
        return self.test_steps(X)



# ========== MODEL SETUP ==========

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"pytorch: using {device} device")

torch_clf = TempModel().to(device)  # Send the model to the GPU
print("\nModel:")
print(torch_clf)
print("")

torch_adv = TempAdversary().to(device)  # Send the adversary to the GPU
print("\nAdversary:")
print(torch_adv)
print("")

predictor_loss_fn = torch.nn.BCELoss()
predictor_optim = torch.optim.Adam(torch_clf.parameters(), lr=1e-4)

adversary_loss_fn = torch.nn.BCELoss()
adversary_optim = torch.optim.Adam(torch_adv.parameters(), lr=1e-4)





def torch_adversarial_train_step(dataset, predictor, adversary, predictor_loss_fn, adversary_loss_fn, predictor_optim, adversary_optim):
    predictor.train() # This means 'set mode to "training"' (bad nomenclature!)
    adversary.train()
    
    for batch, item in enumerate(dataset.get_training_data()):
        
        X, y, z = dataset.split_labels(item[0]) # Torch dataloader biting us again

        # Send the data to the GPU
        X, y, z = X.to(device), y.to(device), z.to(device)
         
        # Put the features through the model's forward pass
        y_pred = predictor(X)

        adversary_input = torch.cat((y_pred, y), 1)

        z_pred = adversary(adversary_input)


        # Backpass (yangisms)
        predictor_loss = predictor_loss_fn(y_pred, y)
        adversary_loss = adversary_loss_fn(z_pred, z)
        sterile_adversary_loss = adversary_loss.item()

        combined_loss = predictor_loss + (predictor_loss / sterile_adversary_loss) - ALPHA * sterile_adversary_loss

        adversary_optim.zero_grad()
        adversary_loss.backward(retain_graph = True)
        
        predictor_optim.zero_grad()
        combined_loss.backward()

        adversary_optim.step()
        predictor_optim.step()
        
        
        


def torch_adversarial_test_step(dataset, predictor, adversary):
    predictor.eval() # Set model to evaluate
    adversary.eval()

    with torch.no_grad(): # Stop the network from recording gradients

        total_correct = 0
        total_data_items = 1e-1000 # Cheap nonzerodiv

        sensitive_sum = 0
        sensitive_count = 1e-1000
        non_sensitive_sum = 0
        non_sensitive_count = 1e-1000

        zt_eo_states = [0, 1e-1000, 0, 1e-1000] # TP, ALLP, FP, ALLN for z == 1
        zf_eo_states = [0, 1e-1000, 0, 1e-1000] # Ditto for z == 0

        for batch, item in enumerate(dataset.get_testing_data()):

            X, y, z = dataset.split_labels(item[0]) # Torch dataloader biting us again

            # Send the data to the GPU
            X, y, z = X.to(device), y.to(device), z.to(device)

            y_pred = predictor(X)

            z_pred = adversary(torch.cat((y_pred, y), 1))


            # Indep calc
            total_correct += (y_pred.round() == y).type(torch.float).sum().item() # Phew
            total_data_items += X.shape[0]
            sensitive_preds = torch.where(z == 1, y_pred.round(), torch.tensor([0.]).to(y_pred.device))
            sensitive_sum += sensitive_preds.sum().item()
            batch_sens_count = torch.count_nonzero(z).item()
            sensitive_count += batch_sens_count
            non_sensitive_preds = torch.where(z == 0, y_pred.round(), torch.tensor([0.]).to(y_pred.device))
            non_sensitive_sum += non_sensitive_preds.sum().item()
            non_sensitive_count += len(z) - batch_sens_count

            # EO calc
            y_and_pred = torch.cat((y, y_pred.round()), 1)
            zt_ys = torch.where(z == 1, y_and_pred, torch.tensor([0.]).to(y.device))
            zf_ys = torch.where(z == 0, y_and_pred, torch.tensor([0.]).to(y.device))

            eq_odds_helper(zt_ys, zt_eo_states)
            eq_odds_helper(zf_ys, zf_eo_states)

        # Compute Independence
        sens_pred_accept = sensitive_sum / sensitive_count
        non_sens_pred_accept = non_sensitive_sum / non_sensitive_count
        independence = sens_pred_accept / non_sens_pred_accept
        independence = 0.0 if math.isnan(independence) else independence

        # Compute EO violation
        sens_true_positive_rate = zt_eo_states[0] / zt_eo_states[1]
        sens_false_positive_rate = zt_eo_states[2] / zt_eo_states[3]
        non_sens_true_positive_rate = zf_eo_states[0] / zf_eo_states[1]
        non_sens_false_positive_rate = zf_eo_states[2] / zf_eo_states[3]
        eo_false = abs(sens_false_positive_rate - non_sens_false_positive_rate)
        eo_true = abs(sens_true_positive_rate - non_sens_true_positive_rate)
        max_eo = max(eo_false, eo_true)
        model_eo_max = 0.0 if math.isnan(max_eo) else max_eo
            
    print(f"{total_correct / total_data_items},{independence},{model_eo_max}") # TODO formatted for csv output



def eq_odds_helper(y_and_pred, states):
    """
        Helper-summer for the above. 
        Takes a tensor of truths and predictions across a batch, previously split by sensitive attribute, and updates
        the [TP, ALLP, FP, ALLN] listuple passed in.
    """

    batch_all_positives = y_and_pred[:, :1].sum().item()
    states[1] += batch_all_positives                        # Update ALLP
    states[3] += y_and_pred.shape[0] - batch_all_positives  # Update ALLN

    states[0] += torch.logical_and(y_and_pred[:,:1] == 1, y_and_pred[:,1:] == 1).type(torch.float).sum().item()
    states[2] += torch.logical_and(y_and_pred[:,:1] == 0, y_and_pred[:,1:] == 1).type(torch.float).sum().item()   



# Epoch Loop
print("epoch,accuracy,independence,eo_violation")
for epoch in range(1024):
    print(f"{epoch},",end='')
    torch_adversarial_train_step(task_dataset, torch_clf, torch_adv, predictor_loss_fn, adversary_loss_fn, predictor_optim, adversary_optim)
    torch_adversarial_test_step(task_dataset, torch_clf, torch_adv)

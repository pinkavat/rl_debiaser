"""
    target_nn_binary.py

    Target model for DDPG-RL-like debiaser. Designed for 'adult' dataset, but
    should support binary-label/binary-fairness classification tasks in general.
"""

import torch
import math # for isnan below

class RLTarget():
    
    def __init__(self, dataset, device_override = None, hidden_layers = 50, learning_rate = 1e-5, parameter_path = 'params_target.pt'):
        
        self.dataset = dataset
        self.device = device_override if device_override else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Target model loaded on {self.device} device")

        self.stale = True           # Test metrics are computed upon request.
        self.model_accuracy = 0.0
        self.model_independence = 0.0
        self.model_eo_max = 0.0
        self.parameter_path = parameter_path


        # ========== MODEL CORE SETUP ==========

        # Model core
        class PrimitiveClf(torch.nn.Module):
            def __init__(self, n_in, n_out, n_hidden):
                super().__init__()
                
                self.steps = torch.nn.Sequential(
                    torch.nn.Linear(n_in, n_hidden),
                    torch.nn.ReLU(),
                    torch.nn.Linear(n_hidden, n_out),
                    torch.nn.Sigmoid()
                )

            def forward(self, X):
                return self.steps(X)

        # Attempt to load parameters from file
        try:
            self.model = PrimitiveClf(dataset.n_x, dataset.n_y, hidden_layers)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            self.loss_fn = torch.nn.BCELoss()

            self.model.load_state_dict(torch.load(parameter_path))
            print("Target parameters loaded from", parameter_path)
            self.model.to(self.device)

        except:

            # Reset and retrain
            retraining_data = dataset.get_retraining_data()

            assert retraining_data is not None, "Don't have parameters and couldn't get retraining data from dataset"

            self.model = PrimitiveClf(dataset.n_x, dataset.n_y, hidden_layers).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            self.loss_fn = torch.nn.BCELoss() # TODO duplicate code

            print("Couldn't load target parameters from", parameter_path, ", retraining target from scratch:")

            # Retrain from scratch
            for epoch in range(10): # TODO a non-hyperparam but ought to still vary
                print(f"\tEpoch {epoch}:")
                self.train(retraining_data)
                print(f"\t\tTest accuracy: {self.get_accuracy()}\n\t\tTest Independence: {self.get_independence()}")
            
            
            # Save classifier
            torch.save(self.model.state_dict(), parameter_path)
            print("Retraining complete; parameters saved at", parameter_path) 


    def train(self, training_data):
        """ Train the target on the given training data. The first part of the environment's reaction to the agent. """

        self.stale = True # The model's changed, so invalidate the test passes.

        self.model.train() # Set for training

        for data_item in training_data:
            if isinstance(data_item, list):
                data_item = data_item[0] # TODO torch dataloader problem

            # Split the labels off
            X, y, z = self.dataset.split_labels(data_item)
            X, y, z = X.to(self.device), y.to(self.device), z.to(self.device)

            y_pred = self.model(X)

            loss = self.loss_fn(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
    def reset(self):
        """ Resets the victim to its initial state as specified in the parameter file. """
        self.model.load_state_dict(torch.load(self.parameter_path))
        self.stale = True


    def get_accuracy(self):
        """ Returns the model's mean accuracy on the test set. """

        if self.stale:
            self.__run_tests()

        return self.model_accuracy


    def get_independence(self):
        """
            Returns P(y'=1 | z=1) / P(y'=1 | z=0)
            Binary Acceptance equality, after Barocas et al.
            The closer to 1, the better.
        """

        if self.stale:
            self.__run_tests()

        return self.model_independence


    def get_max_equalized_odds_violation(self):
        """
            Returns the maximum of
                | FPR(z=1) - FPR(z=0) | and
                | TPR(z=1) - TPR(z=0) |
            Equalized Odds violation; the closer to zero the better.
        """
        if self.stale:
            self.__run_tests()

        return self.model_eo_max



    def __run_tests(self, validation = False):
        """ Internal function that runs a test step to gather metrics. Implicitly invoked by asking for a metric. """

        self.model.eval() # Set for evaluation
        with torch.no_grad():

            total_correct = 0
            total_data_items = 1e-1000 # Cheap nonzerodiv

            sensitive_sum = 0
            sensitive_count = 1e-1000
            non_sensitive_sum = 0
            non_sensitive_count = 1e-1000

            zt_eo_states = [0, 1e-1000, 0, 1e-1000] # TP, ALLP, FP, ALLN for z == 1
            zf_eo_states = [0, 1e-1000, 0, 1e-1000] # Ditto for z == 0


            for data_item in (self.dataset.get_testing_data() if validation else self.dataset.get_training_data()):
                data_item = data_item[0] # TODO torch dataloader problem

                #Split the labels off
                X, y, z = self.dataset.split_labels(data_item)
                X, y, z = X.to(self.device), y.to(self.device), z.to(self.device)

                y_pred = self.model(X)

                total_correct += (y_pred.round() == y).type(torch.float).sum().item() # Phew
                total_data_items += X.shape[0]

                # Better Batch-safe bias metric computation
                # TODO these are binary metric -- it doesn't do multiclass!
                sensitive_preds = torch.where(z == 1, y_pred.round(), torch.tensor([0.]).to(y_pred.device))
                sensitive_sum += sensitive_preds.sum().item()
                batch_sens_count = torch.count_nonzero(z).item()
                sensitive_count += batch_sens_count
                non_sensitive_preds = torch.where(z == 0, y_pred.round(), torch.tensor([0.]).to(y_pred.device))
                non_sensitive_sum += non_sensitive_preds.sum().item()
                non_sensitive_count += len(z) - batch_sens_count

                # Also compute the FPR and TPR for z=0 and z=1
                # TODO overhaul all this stuff with cleaner torch statements
                # For now we duplicate some calcs w.r.t. independence
                y_and_pred = torch.cat((y, y_pred.round()), 1)
                zt_ys = torch.where(z == 1, y_and_pred, torch.tensor([0.]).to(y_pred.device))
                zf_ys = torch.where(z == 0, y_and_pred, torch.tensor([0.]).to(y_pred.device))

                self.__eq_odds_helper(zt_ys, zt_eo_states)
                self.__eq_odds_helper(zf_ys, zf_eo_states)                
                
            
            self.model_accuracy = total_correct / total_data_items

            sens_pred_accept = sensitive_sum / sensitive_count
            non_sens_pred_accept = non_sensitive_sum / non_sensitive_count
            independence = sens_pred_accept / non_sens_pred_accept
            self.model_independence = 0.0 if math.isnan(independence) else independence

            # Compute Equalized Odds maximum
            sens_true_positive_rate = zt_eo_states[0] / zt_eo_states[1]
            sens_false_positive_rate = zt_eo_states[2] / zt_eo_states[3]
            non_sens_true_positive_rate = zf_eo_states[0] / zf_eo_states[1]
            non_sens_false_positive_rate = zf_eo_states[2] / zf_eo_states[3]
            eo_false = abs(sens_false_positive_rate - non_sens_false_positive_rate)
            eo_true = abs(sens_true_positive_rate - non_sens_true_positive_rate)
            max_eo = max(eo_false, eo_true)
            self.model_eo_max = 0.0 if math.isnan(max_eo) else max_eo

        self.stale = False # Metrics are now up-to-date



    def __eq_odds_helper(self, y_and_pred, states):
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

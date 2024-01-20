"""
    compas_dataset_handler.py

    TODO Document
"""

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler



class COMPASDataset():


    def __init__(self, path, train_batch_size, test_batch_size):
        """ Set up the dataloaders and spec. """
    
        # Set up dataloaders
        
        # Load from file into pandas dataframes
        # TODO train-test split parametrize; TODO Cross-validation?
        X_train, X_test, y_train, y_test, z_train, z_test = COMPASDataset.import_data(path, COMPASDataset.sens_attr_sub_sex) # TODO sens attr parametrize

        # Split the training data into data used to train the RL debiaser and data used to evaluate the target's fairness (to generate environmental reward)
        # TODO: this is a major concern -- how much data is enough (stats power question) where do we get it (here? maybe no good) how do we chop it to keep it
        # statistically viable (stats question / crossval question)
        X_train, X_target_eval, y_train, y_target_eval, z_train, z_target_eval = train_test_split(X_train, y_train, z_train, train_size = 0.8, test_size = 0.2)
        X_test, X_target_eval_test, y_test, y_target_eval_test, z_test, z_target_eval_test = train_test_split(X_test, y_test, z_test, test_size = 0.8) # TODO
        
        self.train_data_loader = COMPASDataset.import_to_torch_dataloader(X_train, y_train, z_train, train_batch_size)
        self.test_data_loader = COMPASDataset.import_to_torch_dataloader(X_test, y_test, z_test, test_batch_size)
        self.target_eval_data_loader = COMPASDataset.import_to_torch_dataloader(X_target_eval, y_target_eval, z_target_eval, test_batch_size)
        self.target_eval_test_data_loader = COMPASDataset.import_to_torch_dataloader(X_target_eval_test, y_target_eval_test, z_target_eval_test, test_batch_size)

        # Set shapes etc.
        self.n_x = list(self.train_data_loader.dataset.__getitem__(0)[0].shape)[0] - 2 # TODO note we're subtracting the label count here
        self.n_y = 1
        self.n_z = 1
        self.critic_loss_fn = torch.nn.MSELoss()


    # ========== DATASET ITERABLES ==========

    def get_training_data(self):
        """ Get an iterable training data set. Hands back a pytorch DataLoader. """
        return self.train_data_loader


    def get_tuning_data(self):
        """ Get an iterable testing data set. Hands back a pytorch DataLoader. """
        return self.test_data_loader

    def get_target_eval_data(self):
        """ TODO """
        return self.target_eval_data_loader


    # TODO target eval data loaders?



    # ========= DATA PROPERTIES / AUXILIARIES ==========
    


    # ========== DATA LABELLING ==========

    def split_labels(self, data):
        # TODO unsqueezed; Better, so long as we remember to be consistent.
        return (data[:, :self.n_x], data[:, -(self.n_y + self.n_z):-self.n_z], data[:, -self.n_z:])


    def attach_labels(self, X, y, z):
        # TODO unsqueezed; Better, so long as we remember to be consistent.
        # return torch.cat((torch.cat((X, torch.reshape(y, (len(y), 1))), 1), torch.reshape(z, (len(z), 1))), 1)
        return torch.cat((X, y, z), 1) 


    # ========== ACTOR SPEC ==========

    def get_actor_input_n(self):
        return self.n_x + self.n_y + self.n_z


    def get_actor_output_n(self):
        return self.n_x


    # ========== CRITIC SPEC ==========

    # TODO: old model
    #def get_critic_input_n(self):
    #    return self.n_x + self.n_y + self.n_z
    def get_critic_input_n(self):
        return 2 * (self.n_x + self.n_y + self.n_z)


    def get_critic_output_n(self):
        return 1


    def critic_loss(self, y_true, y_pred):
        return self.critic_loss_fn(y_true, y_pred)


    # ========== TARGET SPEC ==========

    def get_target_input_n(self):
        return self.n_x


    def get_target_output_n(self):
        return self.n_y

   


    # RETROFITTED ADULT IMPORTING CODE FROM OLDER SYSTEM
    # TODO: sensitive attribute not right for multiclass sensitivity --> sensitivity is a onehot thing at present, should be true multiclass

    @staticmethod
    def import_data(path, sensitive_attribute_sub):
        """
            Generate train/test datasets from file and sensitive attribute splitter function.
        """

        """
        Index(['Unnamed: 0', 'id', 'name', 'first', 'last', 'compas_screening_date',
               'sex', 'dob', 'age', 'age_cat', 'race', 'juv_fel_count', 'decile_score',
               'juv_misd_count', 'juv_other_count', 'priors_count',
               'days_b_screening_arrest', 'c_jail_in', 'c_jail_out', 'c_case_number',
               'c_offense_date', 'c_arrest_date', 'c_days_from_compas',
               'c_charge_degree', 'c_charge_desc', 'is_recid', 'r_case_number',
               'r_charge_degree', 'r_days_from_arrest', 'r_offense_date',
               'r_charge_desc', 'r_jail_in', 'r_jail_out', 'violent_recid',
               'is_violent_recid', 'vr_case_number', 'vr_charge_degree',
               'vr_offense_date', 'vr_charge_desc', 'type_of_assessment',
               'decile_score.1', 'score_text', 'screening_date',
               'v_type_of_assessment', 'v_decile_score', 'v_score_text',
               'v_screening_date', 'in_custody', 'out_custody', 'priors_count.1',
               'start', 'end', 'event', 'two_year_recid', 'length_of_stay',
               'score_factor', 'y_pred'],
              dtype='object')

        Alex was using:
                        "sex",
                        "age",
                        "age_cat",
                        "race",
                        "c_charge_degree",
                        "c_charge_desc",
                        "juv_fel_count",
                        "juv_misd_count",
                        "juv_other_count",
                        "priors_count",
                        "two_year_recid",

            TODO TODO TODO:
                NO DECILE SCORE
                TWO YEAR RECID AS LABEL!!!!!!!!!!!!!!!!!!!!!!!!!!!
                
        """


        # Parse file    
        # TODO hardcode drop sex, hardcoded add sensitive (OVERHAUL OVERHAUL)
        # TODO first param is what COMPAS thought...
        feature_names = ['decile_score','age_cat','race','juv_fel_count','juv_misd_count','juv_other_count','priors_count','c_charge_degree','sensitive']
        
        data_raw = pd.read_csv(path, na_values="?", skipinitialspace=True)


        # Flag and drop sensitive attribute
        data_raw = sensitive_attribute_sub(data_raw)


        # Train-test split the data
        y = data_raw['is_recid']
        X = data_raw[feature_names]

        # One-hot encoding
        X = pd.get_dummies(X)

        # Scaling (pop off the sensitive attr then pop back on so it's not scaled)
        z_temp = X['sensitive']
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns = X.columns)
        X['sensitive'] = z_temp

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)


        # Extract the sensitive attribute again (it was packaged for the split, as we don't have multi-target splitting)
        z_train = X_train['sensitive']
        X_train = X_train.drop('sensitive', axis=1)

        z_test = X_test['sensitive']
        X_test = X_test.drop('sensitive', axis=1)

        
        return X_train, X_test, y_train, y_test, z_train, z_test



    @staticmethod
    def sens_attr_sub_sex(df):
        """ Assigns sensitive attribute tag based on whether the sex attribute is 'Female'. """
        df['sensitive'] = df['sex'] == 'Female'
        return df.drop('sex', axis=1)


    @staticmethod
    def import_to_torch_dataloader(X_pd, y_pd, z_pd, batch_size):
        """ Converts the X, y, and z pandas dataframes produced by import_data into a pytorch dataloader. """

        # TODO: concatenation should really be done higher up.

        X = torch.Tensor(X_pd.values)#.unsqueeze(1)
        y = torch.Tensor(y_pd.values).unsqueeze(1) # torch.tensor =/= torch.Tensor (blimey)
        z = torch.Tensor(z_pd.values).unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(torch.cat((X, y, z), 1))
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True) # TODO still necessary to drop ragged batches?

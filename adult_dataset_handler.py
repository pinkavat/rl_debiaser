"""
    adult_dataset_handler.py

    Data Loader and Data Specifications for the Adult Dataset debiasing task.
"""

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler



class AdultDataset():


    def __init__(self, path, train_batch_size, test_batch_size):
        """ Set up the dataloaders and spec. """
    
        # Set up dataloaders
        
        # Load from file into pandas dataframes
        # TODO train-test split parametrize; TODO Cross-validation?
        X_train, X_test, y_train, y_test, z_train, z_test = AdultDataset.import_data(path, AdultDataset.sens_attr_sub_sex) # TODO sens attr parametrize

        # Split the training data into data used to train the RL debiaser and data used to evaluate the target's fairness (to generate environmental reward)
        # TODO: this is a major concern -- how much data is enough (stats power question) where do we get it (here? maybe no good) how do we chop it to keep it
        # statistically viable (stats question / crossval question)
        X_train, X_target_eval, y_train, y_target_eval, z_train, z_target_eval = train_test_split(X_train, y_train, z_train, train_size = 0.2, test_size = 0.1)
        X_test, X_target_eval_test, y_test, y_target_eval_test, z_test, z_target_eval_test = train_test_split(X_test, y_test, z_test, test_size = 0.8) # TODO
        
        self.train_data_loader = AdultDataset.import_to_torch_dataloader(X_train, y_train, z_train, train_batch_size)
        self.test_data_loader = AdultDataset.import_to_torch_dataloader(X_test, y_test, z_test, test_batch_size)
        self.target_eval_data_loader = AdultDataset.import_to_torch_dataloader(X_target_eval, y_target_eval, z_target_eval, test_batch_size)
        self.target_eval_test_data_loader = AdultDataset.import_to_torch_dataloader(X_target_eval_test, y_target_eval_test, z_target_eval_test, test_batch_size)

        # Set shapes etc.
        self.n_x = list(self.train_data_loader.dataset.__getitem__(0)[0].shape)[0] - 2 # TODO note we're subtracting the label count here
        self.n_y = 1
        self.n_z = 1
        self.n_q = 2 # TODO: SEPARATE OBSERVATION AND ACTION SIZES WITH CORRECT NOMENCLATURE RRRRRRRRGH
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
        return (data[:, :self.n_x], data[:, -(self.n_y + self.n_z + self.n_q):-(self.n_z+self.n_q)], data[:, -(self.n_z + self.n_q):-self.n_q], data[:, -self.n_q:])


    def attach_labels(self, X, y, z, q):
        # TODO unsqueezed; Better, so long as we remember to be consistent.
        # return torch.cat((torch.cat((X, torch.reshape(y, (len(y), 1))), 1), torch.reshape(z, (len(z), 1))), 1)
        return torch.cat((X, y, z, q), 1) 


    # ========== ACTOR SPEC ==========

    # TODO TODO: IN THE NEXT ONE, ACTORS KNOW WHAT's WHAT SO THESE FUNCTIONS SHOULD BE IN TERMS OF X, Y, Z size etc!

    def get_actor_input_n(self):
        return self.n_x + self.n_y + self.n_z + self.n_q


    def get_actor_output_n(self):
        return self.n_x


    # ========== CRITIC SPEC ==========


    def get_critic_input_n(self):
        return 2 * (self.n_x + self.n_y + self.n_z + self.n_q)


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
            Returns tuple:
                X_train -- onehot encoded and scaled features, sans sensitive attribute and target.
                X_test
                y_train -- boolean whether income was OVER 50K.
                y_test
                z_train -- boolean set by sensitive_attribute_sub from the sensitive attribute.
                z_test
        """

        # Parse file    
        column_names = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
            "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
            "hours_per_week", "country", "income_cat"]

        integer_features = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss",
            "hours_per_week"]

        categorical_features = ["workclass", "education", "marital_status", "occupation", "relationship",
            "race", "sex", "country"]
        
        data_raw = pd.read_csv(path, names = column_names, na_values="?", skipinitialspace=True)


        # Flag and drop sensitive attribute
        data_raw = sensitive_attribute_sub(data_raw)


        # Train-test split the data
        y = data_raw['income_cat']
        y = y.map(lambda s : not (s == '<=50K' or s == '<=50K.'))
        X = data_raw.drop('income_cat', axis=1)

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
    def sens_attr_sub_race(df):
        """ Assigns sensitive attribute tag based on whether the race attribute is 'Black'. """
        df['sensitive'] = df['race'] == 'Black'
        return df.drop('race', axis=1)


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

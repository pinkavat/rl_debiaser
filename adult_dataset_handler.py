"""
    adult_dataset_handler.py

    Data Loader and Data Specifications for the Adult Dataset debiasing task.
"""

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



class AdultDataset():


    def __init__(self, path, train_size = 0.9, test_size = 0.1, train_batch_size = 32, test_batch_size = 32):
        """ Set up the dataloaders and spec. """
    
        # Set up dataloaders
        self.data_path = path
        
        # Load from file into pandas dataframes
        # TODO Cross-validation?
        X_train, X_test, y_train, y_test, z_train, z_test = AdultDataset.import_data(path, AdultDataset.sens_attr_sub_sex, train_size, test_size) # TODO sens attr parametrize
        
        self.train_data_loader = AdultDataset.import_to_torch_dataloader(X_train, y_train, z_train, train_batch_size)
        self.test_data_loader = AdultDataset.import_to_torch_dataloader(X_test, y_test, z_test, test_batch_size)

        # Set shapes etc.
        self.n_x = list(self.train_data_loader.dataset.__getitem__(0)[0].shape)[0] - 2 # TODO note we're subtracting the label count here
        self.n_y = 1
        self.n_z = 1
        self.data_item_size = self.n_x + self.n_y + self.n_z
        self.critic_loss_fn = torch.nn.MSELoss() # TODO ?


    # ========== DATASET ITERABLES ==========

    def get_training_data(self):
        """ Get an iterable training data set. Hands back a pytorch DataLoader. """
        return self.train_data_loader


    def get_testing_data(self):
        """ Get an iterable testing data set. Hands back a pytorch DataLoader. """
        return self.test_data_loader

    def get_retraining_data(self):
        """ In the event that the target hasn't been trained yet, create just for it a bespoke dataset. """
        # TODO: separation of validation concern.
        X_retrain, _, y_retrain, _, z_retrain, _ = AdultDataset.import_data(self.data_path, AdultDataset.sens_attr_sub_sex, 0.8, 0.2) # Sens attr no-care
        return AdultDataset.import_to_torch_dataloader(X_retrain, y_retrain, z_retrain, 32)


    # ========= DATA PROPERTIES / AUXILIARIES ==========

    def split_labels(self, data):
        return (data[:, :self.n_x], data[:, -(self.n_y + self.n_z):-(self.n_z)], data[:, -(self.n_z):])


   
    # TODO move?
    def critic_loss(self, y_true, y_pred):
        return self.critic_loss_fn(y_true, y_pred)



    # RETROFITTED ADULT IMPORTING CODE FROM OLDER SYSTEM
    # TODO: sensitive attribute not right for multiclass sensitivity --> sensitivity is a onehot thing at present, should be true multiclass

    @staticmethod
    def import_data(path, sensitive_attribute_sub, train_size_, test_size_):
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

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size_, test_size = test_size_, random_state = 42)


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
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=False) # TODO still necessary to drop ragged batches?

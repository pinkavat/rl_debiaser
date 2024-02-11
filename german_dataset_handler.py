"""
    german_dataset_handler.py

    Data Loader and Data Specifications for the German Credit debiasing task.
"""

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



class GermanDataset():


    def __init__(self, path, train_size = 0.7, test_size = 0.3, train_batch_size = 32, test_batch_size = 32):
        """ Set up the dataloaders and spec. """
    
        # Set up dataloaders
        self.data_path = path
        
        # Load from file into pandas dataframes
        X_train, X_test, y_train, y_test, z_train, z_test = self.import_data(path, self.sens_attr_sub_sex, train_size, test_size) # TODO sens attr parametrize
        
        self.train_data_loader = self.import_to_torch_dataloader(X_train, y_train, z_train, train_batch_size)
        self.test_data_loader = self.import_to_torch_dataloader(X_test, y_test, z_test, test_batch_size)

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
        X_retrain, _, y_retrain, _, z_retrain, _ = self.import_data(self.data_path, self.sens_attr_sub_sex, 0.8, 0.2) # Sens attr no-care
        return self.import_to_torch_dataloader(X_retrain, y_retrain, z_retrain, 32)


    # ========= DATA PROPERTIES / AUXILIARIES ==========

    def split_labels(self, data):
        return (data[:, :self.n_x], data[:, -(self.n_y + self.n_z):-(self.n_z)], data[:, -(self.n_z):])


   
    # TODO move?
    def critic_loss(self, y_true, y_pred):
        return self.critic_loss_fn(y_true, y_pred)



    # RETROFITTED ADULT IMPORTING CODE FROM OLDER SYSTEM
    # TODO: sensitive attribute not right for multiclass sensitivity --> sensitivity is a onehot thing at present, should be true multiclass
    def import_data(self, path, sensitive_attribute_sub, train_size_, test_size_):
        """
            Generate train/test datasets from file and sensitive attribute splitter function. 
        """

        # Parse file

        column_names = ['checking_account_status', 'duration', 'credit_history', 'purpose', 'amount', 'savings', 'employment', 'installment_rate', 'sex_and_status', 'guarantors', 'present_residence_since', 'property', 'age', 'other_install', 'housing', 'credits', 'job', 'dependents', 'has_telephone', 'foreign_worker', 'risk']

        numerical_features = ['duration', 'amount', 'installment_rate', 'present_residence_since', 'age', 'credits', 'dependents']
        categorical_features = ['checking_account_status', 'credit_history', 'purpose', 'savings', 'employment', 'guarantors', 'property', 'other_install', 'housing', 'job', 'has_telephone', 'foreign_worker']


        data_raw = pd.read_csv(path, names = column_names, delimiter=' ', na_values="?", skipinitialspace=True)


        # Sex is inexplicably tied to marital status in the categorical variable for it, so we have to split it up
        sex_isolator = lambda item : 'Male' if (item['sex_and_status'] == 'A91' or item['sex_and_status'] == 'A93' or item['sex_and_status'] == 'A94') else 'Female'
        m_s_isolator = lambda item : 'single' if (item['sex_and_status'] in ['A93','A95']) else 'other' # What a metric!

        data_raw['sex'] = data_raw.apply(sex_isolator, axis=1)
        data_raw['marital_status'] = data_raw.apply(m_s_isolator, axis=1)
        data_raw.drop('sex_and_status', axis=1)


        # Flag and drop sensitive attribute
        data_raw = sensitive_attribute_sub(data_raw)


        # Extract labels
        y = data_raw['risk'] == 2 # The label's one-indexed (1 == 'good', 2 == 'bad')
        X = data_raw.drop('risk', axis=1)
        z = X['sensitive']
        X = X.drop('sensitive', axis=1)


        # One-hot encoding
        X = pd.get_dummies(X)
        
        # Scaling
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns = X.columns)

        X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(X, y, z, train_size = train_size_, test_size = test_size_, random_state = 42)
        
        return X_train, X_test, y_train, y_test, z_train, z_test



    def sens_attr_sub_sex(self, df):
        """ Assigns sensitive attribute tag based on whether the sex attribute is 'Female'. """
        df['sensitive'] = df['sex'] == 'Female'
        return df.drop('sex', axis=1)


    def import_to_torch_dataloader(self, X_pd, y_pd, z_pd, batch_size):
        """ Converts the X, y, and z pandas dataframes produced by import_data into a pytorch dataloader. """

        # TODO: concatenation should really be done higher up.

        X = torch.Tensor(X_pd.values)#.unsqueeze(1)
        y = torch.Tensor(y_pd.values).unsqueeze(1) # torch.tensor =/= torch.Tensor (blimey)
        z = torch.Tensor(z_pd.values).unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(torch.cat((X, y, z), 1))
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=False) # TODO still necessary to drop ragged batches?

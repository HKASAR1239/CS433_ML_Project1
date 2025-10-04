import os
import numpy as np
import matplotlib as plt
import helpers as hl
import implementations
import DataExploration as DE

dir_path = os.path.dirname(os.path.realpath(__file__)) + '/data/dataset/'

x_train_v0, x_test_v0, y_train, train_ids, test_ids = hl.load_csv_data(dir_path)

x_train = DE.fill_data(x_train_v0,threshold_features=0.8, normalize=False)
x_test = DE.fill_data(x_test_v0,threshold_features=0.8,normalize=False)

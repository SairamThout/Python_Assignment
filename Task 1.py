#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import unittest

# Load the data (Step 1)
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
ideal_data = pd.read_csv('ideal.csv')

# Adjust column names if necessary
x_col_train = 'x'
y_cols_train = ['y1', 'y2', 'y3', 'y4']
x_col_test = 'x'
y_col_test = 'y'
x_col_ideal = 'x'
y_cols_ideal = ideal_data.columns[1:]

# Function to calculate least squares error
def calculate_least_squares(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

# Find the best 4 ideal functions (Step 2)
ideal_errors = []
for column in y_cols_ideal:
    errors = []
    for train_column in y_cols_train:
        error = calculate_least_squares(train_data[train_column], ideal_data[column])
        errors.append(error)
    ideal_errors.append((column, min(errors)))

# Sort by error and select the best 4
ideal_errors.sort(key=lambda x: x[1])
best_ideal_functions = [column for column, error in ideal_errors[:4]]

# Function to map test data to ideal functions (Step 3)
def map_test_data_to_ideal(test_data, ideal_data, best_ideal_functions, train_data):
    mapping = {}
    sqrt_2 = np.sqrt(2)
    for test_index, test_row in test_data.iterrows():
        for ideal_function in best_ideal_functions:
            deviation = np.abs(ideal_data[ideal_function] - test_row[y_col_test])
            max_deviation = max([np.abs(ideal_data[ideal_function] - train_data[train_col]).max() for train_col in y_cols_train])
            if (deviation <= max_deviation * sqrt_2).all():
                mapping[test_index] = (ideal_function, deviation)
                break
    return mapping

# Map the test data (Step 3)
mapping = map_test_data_to_ideal(test_data, ideal_data, best_ideal_functions, train_data)

# Save the mapping with deviation (Step 3)
mapped_test_data = test_data.copy()
mapped_test_data['Ideal_Function'] = [mapping.get(index, (None, None))[0] for index in test_data.index]
mapped_test_data['Deviation'] = [mapping.get(index, (None, None))[1] for index in test_data.index]
mapped_test_data.to_csv('mapped_test_data.csv', index=False)

# Visualization (Step 4)
plt.figure(figsize=(10, 6))
for function in best_ideal_functions:
    plt.plot(ideal_data[x_col_ideal], ideal_data[function], label=f'Ideal: {function}')
plt.scatter(test_data[x_col_test], test_data[y_col_test], label='Test Data', color='red', zorder=5)
plt.legend()
plt.xlabel(x_col_ideal)
plt.ylabel(y_col_test)
plt.title('Best Fit Ideal Functions and Test Data')
plt.show()

# Unit tests (Step 5)
class TestMapping(unittest.TestCase):
    def test_least_squares(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        self.assertEqual(calculate_least_squares(y_true, y_pred), 0)

    def test_best_ideal_functions(self):
        self.assertEqual(len(best_ideal_functions), 4)

    def test_mapping(self):
        self.assertTrue(len(mapping) > 0)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)


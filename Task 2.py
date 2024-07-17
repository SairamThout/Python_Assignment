#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlalchemy as db
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
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

# Define the database models
Base = declarative_base()

class TrainingData(Base):
    __tablename__ = 'training_data'
    id = db.Column(db.Integer, primary_key=True)
    x = db.Column(db.Float)
    y1 = db.Column(db.Float)
    y2 = db.Column(db.Float)
    y3 = db.Column(db.Float)
    y4 = db.Column(db.Float)

class IdealFunction(Base):
    __tablename__ = 'ideal_functions'
    id = db.Column(db.Integer, primary_key=True)
    x = db.Column(db.Float)
    y = db.Column(db.Float)
    
class TestMapping(Base):
    __tablename__ = 'test_mapping'
    id = db.Column(db.Integer, primary_key=True)
    x = db.Column(db.Float)
    y = db.Column(db.Float)
    ideal_function = db.Column(db.String)
    deviation = db.Column(db.Float)

# Create an SQLite database and store the training and ideal functions (Step 2)
engine = db.create_engine('sqlite:///functions.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Insert training data into the database
for index, row in train_data.iterrows():
    session.add(TrainingData(x=row[x_col_train], y1=row[y_cols_train[0]], y2=row[y_cols_train[1]], y3=row[y_cols_train[2]], y4=row[y_cols_train[3]]))
session.commit()

# Insert ideal functions into the database
for index, row in ideal_data.iterrows():
    for col in y_cols_ideal:
        session.add(IdealFunction(x=row[x_col_ideal], y=row[col]))
session.commit()

# Function to calculate least squares error
def calculate_least_squares(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

# Find the best 4 ideal functions (Step 3)
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

# Function to map test data to ideal functions and save the results in the database (Step 4)
def map_test_data_to_ideal(test_data, ideal_data, best_ideal_functions, train_data):
    mapping = {}
    sqrt_2 = np.sqrt(2)
    for test_index, test_row in test_data.iterrows():
        for ideal_function in best_ideal_functions:
            deviation_series = np.abs(ideal_data[ideal_function] - test_row[y_col_test])
            max_deviation = max([np.abs(ideal_data[ideal_function] - train_data[train_col]).max() for train_col in y_cols_train])
            if (deviation_series <= max_deviation * sqrt_2).all():
                deviation = deviation_series.iloc[test_index]
                mapping[test_index] = (ideal_function, deviation)
                session.add(TestMapping(x=test_row[x_col_test], y=test_row[y_col_test], ideal_function=ideal_function, deviation=deviation))
                break
    session.commit()
    return mapping

# Map the test data
mapping = map_test_data_to_ideal(test_data, ideal_data, best_ideal_functions, train_data)

# Save the mapping with deviation (Step 4)
mapped_test_data = test_data.copy()
mapped_test_data['Ideal_Function'] = [mapping.get(index, (None, None))[0] for index in test_data.index]
mapped_test_data['Deviation'] = [mapping.get(index, (None, None))[1] for index in test_data.index]
mapped_test_data.to_csv('mapped_test_data.csv', index=False)

# Step 5: Visualize the results
plt.figure(figsize=(10, 6))
for function in best_ideal_functions:
    plt.plot(ideal_data[x_col_ideal], ideal_data[function], label=f'Ideal: {function}')
plt.scatter(test_data[x_col_test], test_data[y_col_test], label='Test Data', color='red', zorder=5)
plt.legend()
plt.xlabel(x_col_ideal)
plt.ylabel(y_col_test)
plt.title('Best Fit Ideal Functions and Test Data')
plt.show()

# Visualize mapping results
plt.figure(figsize=(10, 6))
for test_index, (ideal_function, deviation) in mapping.items():
    plt.scatter(test_data.loc[test_index, x_col_test], test_data.loc[test_index, y_col_test], label=f'Test Data Mapped to {ideal_function} with deviation {deviation}', zorder=5)
plt.legend()
plt.xlabel(x_col_ideal)
plt.ylabel(y_col_test)
plt.title('Test Data Mapped to Ideal Functions')
plt.show()

# Step 6: Unit tests
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


# In[ ]:





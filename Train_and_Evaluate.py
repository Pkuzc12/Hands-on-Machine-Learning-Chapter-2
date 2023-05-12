import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

DATA_PATH = os.path.join("datasets", "housing")


def load_data(data_path=DATA_PATH):
    csv_path = os.path.join(data_path, "housing.csv")
    return pd.read_csv(csv_path)


rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAtrributeAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        population_per_household = X[:, population_ix]/X[:, household_ix]
        rooms_per_household = X[:, rooms_ix]/X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix]/X[:, rooms_ix]
            return np.c_[X, population_per_household, rooms_per_household, bedrooms_per_room]
        else:
            return np.c_[X, population_per_household, rooms_per_household]


pipeline_num = Pipeline([
    ["impute", SimpleImputer(strategy="median")],
    ["CombinedAtrributeAdd", CombinedAtrributeAdder()],
    ["StandardScaler", StandardScaler()],
])


housing = load_data()
housing["income_cut"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3., 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cut"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
for set_ in [strat_train_set, strat_test_set]:
    set_.drop("income_cut", axis=1, inplace=True)
housing = strat_train_set.copy()
housing_num = housing.drop("ocean_proximity", axis=1, inplace=False)

attri_num = list(housing_num)
attri_cat = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
    ["num", pipeline_num, attri_num],
    ["cat", OneHotEncoder(), attri_cat],
])

housing_prepared = full_pipeline.fit_transform(housing)

# lin_reg = LinearRegression()
# lin_reg.fit(housing_prepared, housing_labels)
# some_data = housing_prepared.iloc[:5]
# some_labels = housing_labels.iloc[:5]
# print(lin_reg.predict(some_data))
# print(some_labels)

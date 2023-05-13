import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

DATA_PATH = os.path.join("datasets", "housing")


def load_data(data_path=DATA_PATH):
    csv_path = os.path.join(data_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_data()
housing["income_cut"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3., 4.5, 6., np.inf])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cut"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
for set_ in [strat_train_set, strat_test_set]:
    set_.drop("income_cut", axis=1, inplace=True)
housing = strat_train_set.copy()

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
global add_bedrooms_per_rooms
add_bedrooms_per_rooms = True


class CombinedAttributeAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.add_bedrooms_per_rooms = add_bedrooms_per_rooms

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        pop_per_house = X[:, population_ix]/X[:, household_ix]
        bedrooms_per_house = X[:, bedrooms_ix]/X[:, household_ix]
        if self.add_bedrooms_per_rooms:
            bedrooms_per_rooms = X[:, bedrooms_ix]/X[:, rooms_ix]
            return np.c_[X, pop_per_house, bedrooms_per_house, bedrooms_per_rooms]
        else:
            return np.c_[X, pop_per_house, bedrooms_per_house]


pipline_num = Pipeline([
    ["impute", SimpleImputer()],
    ["add", CombinedAttributeAdder()],
    ["standard", StandardScaler()],
])

housing_num = housing.drop(["ocean_proximity", "median_house_value"], axis=1, inplace=False)
attri_num = list(housing_num)
attri_cat = ["ocean_proximity"]

pipeline_full = ColumnTransformer([
    ("num", pipline_num, attri_num),
    ("cat", OneHotEncoder(), attri_cat),
])

housing_labels = housing["median_house_value"].copy()
housing_prepared = pipeline_full.fit_transform(housing.drop("median_house_value", axis=1, inplace=False))
if add_bedrooms_per_rooms:
    columns_prepared = attri_num+["pop_per_house", "bedrooms_per_house", "bedrooms_per_rooms"]+['1', '2', '3', '4', '5']
else:
    columns_prepared = attri_num+["pop_per_house", "bedrooms_per_house"]+['1', '2', '3', '4', '5']
housing_prepared = pd.DataFrame(housing_prepared, columns=columns_prepared, index=housing.index)

lin_reg = LinearRegression()
decision_tree_reg = DecisionTreeRegressor()


def show_scores(scores):
    print(scores.mean())
    print(scores.std())


scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
scores = np.sqrt(-scores)
print("LinearRegression")
show_scores(scores)

scores = cross_val_score(decision_tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
scores = np.sqrt(-scores)
print("DecisionTreeRegression")
show_scores(scores)

joblib.dump(lin_reg, "lin_reg.pkl")
joblib.dump(decision_tree_reg, "decision_tree_reg.pkl")

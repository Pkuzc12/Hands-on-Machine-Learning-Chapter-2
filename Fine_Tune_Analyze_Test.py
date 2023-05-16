import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

DATA_PATH = os.path.join("datasets", "housing")


def load_data(data_path=DATA_PATH):
    csv_path = os.path.join(DATA_PATH, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_data()
housing["income_cut"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3., 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cut"]):
    strat_test_set = housing.loc[test_index]
    strat_train_set = housing.loc[train_index]
for set_ in [strat_test_set, strat_train_set]:
    set_.drop("income_cut", axis=1, inplace=True)
housing = strat_train_set.copy()

rooms_ix, bedrooms_ix, pop_ix, household_ix = 3, 4, 5, 6


class CombinedAttributeAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_rooms=True):
        self.add_bedrooms_per_rooms = add_bedrooms_per_rooms

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        pop_per_household = X[:, pop_ix]/X[:, household_ix]
        bedrooms_per_household = X[:, bedrooms_ix]/X[:, household_ix]
        if self.add_bedrooms_per_rooms:
            bedrooms_per_rooms = X[:, bedrooms_ix]/X[:, rooms_ix]
            return np.c_[X, pop_per_household, bedrooms_per_household, bedrooms_per_rooms]
        else:
            return np.c_[X, pop_per_household, bedrooms_per_household]


pipeline_num = Pipeline([
    ["impute", SimpleImputer(strategy="median")],
    ["add", CombinedAttributeAdder()],
    ["standard", StandardScaler()],
])

housing_num = housing.drop(["ocean_proximity", "median_house_value"], axis=1, inplace=False)
attri_num = list(housing_num)
attri_cat = ["ocean_proximity"]

pipeline_full = ColumnTransformer([
    ("num", pipeline_num, attri_num),
    ("cat", OneHotEncoder(), attri_cat),
])

housing_labels = housing["median_house_value"].copy()
columns_new = list(housing_num)+["pop_per_house", "bedrooms_per_house", "bedrooms_per_rooms"]+["1", "2", "3", "4", "5"]
housing_prepred = pd.DataFrame(pipeline_full.fit_transform(housing.drop("median_house_value", axis=1, inplace=False)), columns=columns_new, index=housing.index)

lin_reg = LinearRegression()
dec_tree_reg = DecisionTreeRegressor()


def show_scores(scores):
    print(scores.mean())
    print(scores.std())


scores = cross_val_score(lin_reg, housing_prepred, housing_labels, scoring="neg_mean_squared_error", cv=10)
scores = np.sqrt(-scores)
show_scores(scores)

scores = cross_val_score(dec_tree_reg, housing_prepred, housing_labels, scoring="neg_mean_squared_error", cv=10)
scores = np.sqrt(-scores)
show_scores(scores)

joblib.dump(lin_reg, "lin_reg.pkl")
joblib.dump(dec_tree_reg, "dec_tree_reg.pkl")

para_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, para_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepred, housing_labels)
print(grid_search.best_params_)
print(grid_search.best_estimator_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

joblib.dump(forest_reg, "forest_reg.pkl")

feature_importances = grid_search.best_estimator_.feature_importances_
print(sorted(zip(feature_importances, columns_new), reverse=True))

final_model = grid_search.best_estimator_

final = strat_test_set.copy()
final_prepared = pd.DataFrame(pipeline_full.fit_transform(final.drop(["median_house_value"], axis=1, inplace=False)), columns=columns_new, index=final.index)
final_labels = final["house_median_value"].copy()
final_predictions = final_model.predict(final_prepared)
final_mse = mean_squared_error(final_predictions, final_labels)
final_rmse = np.sqrt(final_mse)

joblib.dump(final_model, "final_model.pkl")
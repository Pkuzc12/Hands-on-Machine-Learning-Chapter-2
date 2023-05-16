import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

DATA_PATH = os.path.join("datasets", "housing")


def load_data(data_path=DATA_PATH):
    csv_path = os.path.join(data_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_data()
housing["income_cut"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3., 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cut"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
for set_ in [strat_train_set, strat_test_set]:
    set_.drop("income_cut", axis=1, inplace=True)
housing = strat_train_set.copy()

rooms_ix, bedrooms_ix, pop_ix, house_ix = 3, 4, 5, 6


class CombinedAttributeAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_rooms=True):
        self.add_bedrooms_per_rooms = add_bedrooms_per_rooms

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        pop_per_house = X[:, pop_ix]/X[:, house_ix]
        bedrooms_per_house = X[:, bedrooms_ix]/X[:, house_ix]
        if self.add_bedrooms_per_rooms:
            bedrooms_per_rooms = X[:, bedrooms_ix]/X[:, rooms_ix]
            return np.c_[X, pop_per_house, bedrooms_per_house, bedrooms_per_rooms]
        else:
            return np.c_[X, pop_per_house, bedrooms_per_house]


pipeline_num = Pipeline([
    ["impute", SimpleImputer(strategy="median")],
    ["add", CombinedAttributeAdder()],
    ["stand", StandardScaler()],
])

housing_num = housing.drop(["median_house_value", "ocean_proximity"], axis=1, inplace=False)
attri_num = list(housing_num)
attri_cat = ["ocean_proximity"]

pipeline_full = ColumnTransformer([
    ("num", pipeline_num, attri_num),
    ("cat", OneHotEncoder(), attri_cat),
])

columns_new = attri_num+["pop_per_house", "bedrooms_per_house", "bedrooms_per_rooms"]+["1", "2", "3", "4", "5"]
housing_labels = housing["median_house_value"].copy()
housing_prepared = pd.DataFrame(pipeline_full.fit_transform(housing.drop("median_house_value", axis=1, inplace=False)), columns=columns_new, index=housing.index)

rand_forest_reg = RandomForestRegressor()
lin_reg = LinearRegression()
tree_reg = DecisionTreeRegressor()


def show_scores(scores):
    print(scores.mean())
    print(scores.std())


scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
scores = np.sqrt(-scores)
show_scores(scores)

scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
scores = np.sqrt(-scores)
show_scores(scores)

para_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

grid_search = GridSearchCV(rand_forest_reg, para_grid, scoring="neg_mean_squared_error", cv=5, return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
print(grid_search.best_estimator_)
print(grid_search.best_params_)
cv_results = grid_search.cv_results_
for scores, params in zip(cv_results["mean_test_score"], cv_results["params"]):
    print(np.sqrt(-scores), params)

feature_importances = grid_search.best_estimator_.feature_importances_
print(sorted(zip(feature_importances, columns_new), reverse=True))

final_model = grid_search.best_estimator_
joblib.dump(final_model, "final_model.pkl")

final = strat_test_set.copy()
final_labels = final["median_house_value"].copy()
final_prepared = pd.DataFrame(pipeline_full.fit_transform(final.drop("median_house_value", axis=1, inplace=False)), columns=columns_new, index=final.index)
final_predicts = final_model.predict(final_prepared)
final_mse = mean_squared_error(final_labels, final_predicts)
final_rmse = np.sqrt(final_mse)
print(final_rmse)

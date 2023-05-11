import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit

DATA_PATH = os.path.join("datasets", "housing")


def load_data(data_path=DATA_PATH):
    csv_path = os.path.join(data_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_data()
housing["income_cut"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cut"]):
    strat_test_set = housing.loc[test_index]
    strat_train_set = housing.loc[train_index]
for set_ in [strat_test_set, strat_train_set]:
    set_.drop("income_cut", axis=1, inplace=True)
housing = strat_train_set.copy()
housing.drop("ocean_proximity", axis=1, inplace=True)
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

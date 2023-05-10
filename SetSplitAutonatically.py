import pandas as pd
import os
from sklearn.model_selection import train_test_split

HOUSING_PATH = os.path.join("datasets", "housing")


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_housing_data()
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

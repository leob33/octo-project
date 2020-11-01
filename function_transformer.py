import pandas as pd
import numpy as np

def categorize_age(X):
    X['Age_cat'] = pd.cut(X.Age, (18, 25, 35, 60, 120), labels=['Baby', 'Young', 'Adult', 'Senior'])
    return X


def fill_na(X):
    return X.replace(np.nan, "unknown")


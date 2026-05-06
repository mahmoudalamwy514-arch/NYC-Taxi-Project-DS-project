from sklearn.utils import resample
import pandas as pd

def balance_data(df, target):
    majority = df[df[target] == 0]
    minority = df[df[target] == 1]

    minority_up = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=42
    )

    balanced = pd.concat([majority, minority_up])
    print("Balanced:", balanced[target].value_counts())

    return balanced
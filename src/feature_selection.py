from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor

def select_features(X, y):

    # Remove any NaN safety
    X = X.dropna()
    y = y.loc[X.index]

    # Method 1: SelectKBest
    kbest = SelectKBest(score_func=f_regression, k=min(5, X.shape[1]))
    kbest.fit(X, y)
    kbest_features = X.columns[kbest.get_support()]

    # Method 2: Random Forest
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X, y)

    importances = rf.feature_importances_
    rf_features = X.columns[importances.argsort()[-5:]]

    print("KBest:", list(kbest_features))
    print("RF:", list(rf_features))

    return list(kbest_features)
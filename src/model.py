from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

def train_model(df, features, target):

    # safety: reduce memory issues
    df = df.dropna()
    df = df.sample(min(50000, len(df)), random_state=42)

    X = df[features]
    y = df[target]

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # lighter + safer model (important for your error)
    model = RandomForestRegressor(
        n_estimators=50,   # reduced from 100
        max_depth=12,      # prevent memory explosion
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test


def save_model(model, path="model.pkl"):
    joblib.dump(model, path)


def load_model(path="model.pkl"):
    return joblib.load(path)
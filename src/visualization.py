import matplotlib.pyplot as plt

def plot_duration(df):
    plt.hist(df["trip_duration"], bins=50)
    plt.title("Trip Duration Distribution")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Count")
    plt.show()


def plot_distance_vs_duration(df):
    plt.scatter(df["distance_km"], df["trip_duration"])
    plt.title("Distance vs Trip Duration")
    plt.xlabel("Distance (km)")
    plt.ylabel("Trip Duration")
    plt.show()
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

def zad_1(data):
    print("=== HEAD ===")
    print(data.head())

    print("\n=== INFO ===")
    print(data.info())

    print("\n=== BRAKI DANYCH ===")
    print(data.isnull().sum())

    print("\n=== STATYSTYKI OPISOWE ===")
    print(data.describe())

    height = data["Height(Inches)"]
    weight = data["Weight(Pounds)"]

    print("\n=== HEIGHT ===")
    print("Mean:", height.mean())
    print("Median:", height.median())
    print("Std:", height.std())
    print("Skewness:", skew(height))
    print("Kurtosis:", kurtosis(height))

    print("\n=== WEIGHT ===")
    print("Mean:", weight.mean())
    print("Median:", weight.median())
    print("Std:", weight.std())
    print("Skewness:", skew(weight))
    print("Kurtosis:", kurtosis(weight))

    plt.figure()
    sns.histplot(height, kde=True)
    plt.title("Rozkład wzrostu")
    plt.savefig(os.path.join("images", "wzrost.png"))
    plt.show()

    plt.figure()
    sns.histplot(weight, kde=True)
    plt.title("Rozkład wagi")
    plt.savefig(os.path.join("images", "waga.png"))
    plt.show()

    plt.figure()
    sns.scatterplot(x=height, y=weight)
    plt.title("Wzrost vs Waga")
    plt.xlabel("Height (inches)")
    plt.ylabel("Weight (pounds)")
    plt.savefig(os.path.join("images", "waga_wzrost.png"))
    plt.show()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import os

os.makedirs("images", exist_ok=True)

path = r"data\SOCR-HeightWeight.csv"
data = pd.read_csv(path)
values = data["Height(Inches)"]

mean = values.mean()
median = values.median()
std = values.std()

q1 = values.quantile(0.25)
q2 = values.quantile(0.50)
q3 = values.quantile(0.75)

skewness = skew(values)
kurt = kurtosis(values)

print("Średnia:", mean)
print("Mediana:", median)
print("Q1:", q1)
print("Q2:", q2)
print("Q3:", q3)
print("Odchylenie standardowe:", std)
print("Skośność:", skewness)
print("Kurtoza:", kurt)

plt.figure(figsize=(8,5))
plt.hist(values, bins=40)
plt.title("Wykres wzrostu ludzi")
plt.xlabel("Wzrost [cal]")
plt.ylabel("Liczba osób")
plt.savefig("images/wzrost_licznosc.png")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x=values)
plt.title("Boxplot wzrostu ludzi")
plt.xlabel("Wzrost [cal]")
plt.savefig("images/boxplot.png")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x="Height(Inches)", y="Weight(Pounds)", data=data)
plt.title("Zależność wzrostu i wagi")
plt.xlabel("Wzrost [cal]")
plt.ylabel("Waga [funt]")
plt.savefig("images/wzrost_waga.png")
plt.show()
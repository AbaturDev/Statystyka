import os
import zad1
import zad2
import zad3
import zad4
import zad5
import pandas as pd

os.makedirs("images", exist_ok=True)

path = os.path.join("data", "SOCR-HeightWeight.csv")
data = pd.read_csv(path)
height = data["Height(Inches)"]
weight = data["Weight(Pounds)"]

zad1.zad_1(data)
pearson = zad2.zad_2(height, weight)
spearman = zad3.zad_3(height, weight, pearson)
zad4.zad_4(pearson, spearman)
zad5.zad_5(data)
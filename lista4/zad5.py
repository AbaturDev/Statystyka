import os
import seaborn as sns
import matplotlib.pyplot as plt

def zad_5(data):
    corr_matrix = data[['Height(Inches)', 'Weight(Pounds)']].corr()
    print("\n=== MACIERZ KORELACJI ===")
    print(corr_matrix)

    plt.figure()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Macierz korelacji")
    plt.savefig(os.path.join("images", "macierz.png"))
    plt.show()

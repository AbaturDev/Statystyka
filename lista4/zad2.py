import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from common import interpret_corr

def zad_2(height, weight):
    corr, p = pearsonr(height, weight)

    print("\n=== PEARSON ===")
    print("r:", corr)
    print("p-value:", p)

    print("\n=== INTERPRETACJA ===")
    print("Siła:", interpret_corr(corr))
    print("Kierunek:", "dodatni" if corr > 0 else "ujemny")

    if p < 0.05:
        print("Zależność istotna statystycznie")
    else:
        print("Zależność nieistotna statystycznie")

    if abs(corr) > 0.3:
        print("Zależność ma znaczenie praktyczne")
    else:
        print("Zależność słaba w sensie praktycznym")

    plt.figure()
    sns.regplot(x=height, y=weight, scatter_kws={'alpha':0.5})
    plt.title(f"Pearson r={corr:.3f}")
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.savefig(os.path.join("images", "pearson.png"))
    plt.show()

    return {
        "corr": corr,
        "p": p
    }
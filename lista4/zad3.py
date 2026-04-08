from scipy.stats import spearmanr
from common import interpret_corr

def zad_3(height, weight, pearson):
    corr, p = spearmanr(height, weight)

    print("\n=== SPEARMAN ===")
    print("rho:", corr)
    print("p-value:", p)

    print("\n=== INTERPRETACJA ===")
    print("Siła:", interpret_corr(corr))
    print("Kierunek:", "dodatni" if corr > 0 else "ujemny")

    print("\n=== PORÓWNANIE Z PEARSONEM ===")
    print(f"Pearson: {pearson['corr']:.4f}")
    print(f"Spearman: {corr:.4f}")

    diff = abs(pearson["corr"] - corr)

    if diff < 0.05:
        print("Wyniki bardzo podobne → zależność liniowa, brak istotnych outlierów")
    else:
        print("Różnice sugerują nieliniowość lub obecność outlierów")

    print("\n=== WNIOSKI ===")
    print("Spearman opiera się na rangach, więc:")
    print("- jest odporny na wartości odstające")
    print("- wykrywa zależności monotoniczne (nie tylko liniowe)")

    return {
        "corr": corr,
        "p": p
    }
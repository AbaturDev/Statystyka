def zad_4(pearson, spearman):
    alpha = 0.05

    print("\n=== TEST ISTOTNOŚCI ===")

    print("\nHipotezy:")
    print("H0: brak korelacji (r = 0)")
    print("H1: istnieje korelacja (r ≠ 0)")

    print("\nPoziom istotności α =", alpha)

    print("\n=== Pearson ===")
    print("p-value:", pearson["p"])

    if pearson["p"] < alpha:
        print("Odrzucamy H0 – korelacja istotna statystycznie")
    else:
        print("Brak podstaw do odrzucenia H0")

    print("\n=== Spearman ===")
    print("p-value:", spearman["p"])

    if spearman["p"] < alpha:
        print("Odrzucamy H0 – korelacja istotna statystycznie")
    else:
        print("Brak podstaw do odrzucenia H0")
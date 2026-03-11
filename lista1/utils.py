import numpy as np

def stats(data):
    return np.mean(data), np.std(data)

def print_stats(title, mean, std):
    print("\n")
    print(title)
    print("Średnia:", mean)
    print("Odchylenie:", std)
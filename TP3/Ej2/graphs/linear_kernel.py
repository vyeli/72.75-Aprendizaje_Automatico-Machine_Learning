import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("../output/sky_vs_cow_C_linear.csv")

C_values = df['C'].unique()

grouped_by_C = df.groupby('C')

mean_times = []
err_times = []

mean_accuracies = []
err_accuracies = []

for C, grouped in grouped_by_C:
    mean_times.append(grouped['time'].mean())
    err_times.append(grouped['time'].std())

    mean_accuracy = (grouped['tp'].mean() + grouped['tn'].mean()) / (grouped['tp'].mean() + grouped['tn'].mean() + grouped['fp'].mean() + grouped['fn'].mean())
    mean_accuracies.append(mean_accuracy)
    print(mean_accuracy)
    err_accuracies.append((grouped['tp'].std() + grouped['tn'].std()) / (grouped['tp'].mean() + grouped['tn'].mean() + grouped['fp'].mean() + grouped['fn'].mean()))

plt.errorbar(C_values, mean_times, yerr=err_times, marker='o', linestyle="dashed")

plt.ylabel("Tiempo de ejecución [s]")
plt.xlabel("C")
plt.savefig("../output/sky_vs_cow_linear_time.png")
plt.show()
plt.clf()

plt.errorbar(C_values, mean_accuracies, yerr=err_accuracies, marker='o', linestyle="dashed")

plt.ylabel("Accuracy")
plt.xlabel("C")
plt.savefig("../output/sky_vs_cow_linear_accuracy.png")
plt.show()
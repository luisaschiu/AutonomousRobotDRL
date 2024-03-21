import pandas as pd
import matplotlib.pyplot as plt

training_4 = pd.read_csv("results_4x4_for_100/training_data.csv")
gameplay_4 = pd.read_csv("results_4x4_for_100/gameplay_data.csv")

training_8 = pd.read_csv("results_8x8_for_200/training_data.csv")
gameplay_8 = pd.read_csv("results_8x8_for_200/gameplay_data.csv")

print("MAE")
mae_gameplay_4 = sum(gameplay_4["Steps"] - 6) / 100
print(mae_gameplay_4)

print("MSE")
mae_squared_4 = sum((gameplay_4["Steps"] - 6) ** 2) / 100
print(mae_squared_4)

print("MAE")
mae_gameplay_8 = sum(gameplay_8["Steps"] - 14) / 100
print(mae_gameplay_8)

print("MSE")
mae_squared_8 = sum((gameplay_8["Steps"] - 14) ** 2) / 100
print(mae_squared_8)


plt.plot(training_4["Episode"], training_4["Steps"])
plt.xlabel("Episode #")
plt.ylabel("Number of Steps")
plt.title("Training - 4 x 4")
plt.savefig("training_4.png")
plt.figure()

plt.plot(gameplay_4["Episode"], gameplay_4["Steps"])
plt.xlabel("Episode #")
plt.ylabel("Number of Steps")
plt.title("Gameplay - 4 x 4")
plt.savefig("gameplay_4.png")
plt.figure()

plt.plot(training_8["Episode"], training_8["Steps"])
plt.xlabel("Episode #")
plt.ylabel("Number of Steps")
plt.title("Training - 8 x 8")
plt.savefig("training_8.png")
plt.figure()

plt.plot(gameplay_8["Episode"], gameplay_8["Steps"])
plt.xlabel("Episode #")
plt.ylabel("Number of Steps")
plt.title("Gameplay - 8 x 8")
plt.savefig("gameplay_8.png")
plt.figure()
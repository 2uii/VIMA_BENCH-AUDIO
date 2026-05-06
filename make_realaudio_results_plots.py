import matplotlib.pyplot as plt
import csv

# -----------------------
# Main model comparison
# -----------------------
model_results = [
    ("No-audio baseline", 0.030058),
    ("Old Audio-VIMA", 0.029948),
    ("Real-audio Audio-VIMA", 0.026178),
]

with open("model_comparison_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Best Test Loss"])
    writer.writerows(model_results)

plt.figure(figsize=(8, 5))
plt.bar([x[0] for x in model_results], [x[1] for x in model_results])
plt.ylabel("Best Test Loss")
plt.title("Model Comparison")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("model_comparison_results.png")
plt.close()

# -----------------------
# Audio ablation
# -----------------------
ablation_results = [
    ("Clean audio", 0.026178),
    ("Shuffled audio", 0.026709),
    ("Zero audio", 0.030784),
]

with open("audio_ablation_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Condition", "Test Loss"])
    writer.writerows(ablation_results)

plt.figure(figsize=(8, 5))
plt.bar([x[0] for x in ablation_results], [x[1] for x in ablation_results])
plt.ylabel("Test Loss")
plt.title("Real-Audio Ablation Results")
plt.tight_layout()
plt.savefig("audio_ablation_results.png")
plt.close()

print("Saved:")
print(" - model_comparison_results.csv")
print(" - model_comparison_results.png")
print(" - audio_ablation_results.csv")
print(" - audio_ablation_results.png")

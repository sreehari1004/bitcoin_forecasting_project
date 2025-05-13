import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Model list
models = ['SVR', 'FFNN', 'KNN', 'RT', 'RBFNN', 'BRNN', 'LSTM', 'GRU']

# Initialize data
r2_scores = []
accuracies = []

# Extract metrics from files
for model in models:
    try:
        with open(f"Results/{model}.txt", 'r') as f:
            content = f.read()
            r2_match = re.search(r'R²:\s*([-+]?[0-9]*\.?[0-9]+)', content)
            acc_match = re.search(r'Accuracy:\s*([-+]?[0-9]*\.?[0-9]+)', content)
            if r2_match and acc_match:
                r2_scores.append(float(r2_match.group(1)))
                accuracies.append(float(acc_match.group(1)) / 100.0)  # convert to decimal
            else:
                print(f"Missing data in {model}.txt")
    except FileNotFoundError:
        print(f"File {model}.txt not found.")

# Create DataFrame
df = pd.DataFrame({
    'Model': models,
    'R² Score': r2_scores,
    'Accuracy': accuracies  # already in decimal form
})

# Plot grouped bar chart
x = range(len(models))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar([i - width/2 for i in x], df['R² Score'], width=width, label='R² Score', color='skyblue')
plt.bar([i + width/2 for i in x], df['Accuracy'], width=width, label='Accuracy (as decimal)', color='salmon')

plt.xticks(x, df['Model'], rotation=45)
plt.ylabel("Metric Value (0 to 1)")
plt.title("Model Comparison: R² Score vs Accuracy")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()

# Save the figure
os.makedirs("Results/Plots", exist_ok=True)
plt.savefig("Results/Plots/Model_Comparison_R2_Accuracy.pdf", format='pdf', dpi=600)
plt.show()



# Plot grouped bar chart with enhanced text size
x = range(len(models))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar([i - width/2 for i in x], df['R² Score'], width=width, label='R² Score', color='skyblue', edgecolor='black', linewidth=1.2)
plt.bar([i + width/2 for i in x], df['Accuracy'], width=width, label='Accuracy (as decimal)', color='salmon', edgecolor='black', linewidth=1.2)

plt.xticks(x, df['Model'], rotation=45, fontsize=18, weight='bold')
plt.ylabel("Metric Value (0 to 1)", fontsize=22, weight='bold')
plt.title("Model Comparison: R² Score vs Accuracy", fontsize=28, weight='bold', pad=15)
plt.legend(fontsize=16)
plt.yticks(fontsize=18, weight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the figure
os.makedirs("Results/Plots", exist_ok=True)
plt.savefig("Results/Plots/Model_Comparison_R2_Accuracy.pdf", format='pdf', dpi=600)
plt.show()

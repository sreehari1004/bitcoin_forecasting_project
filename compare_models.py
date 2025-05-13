import matplotlib.pyplot as plt

# Read RT_rmse from file
with open('Results/SVR_rmse.txt', 'r') as file:
    SVR_rmse = float(file.read())
with open('Results/FFNN_rmse.txt', 'r') as file:
    FFNN_rmse = float(file.read())
with open('Results/KNN_rmse.txt', 'r') as file:
    KNN_rmse = float(file.read())
with open('Results/RT_rmse.txt', 'r') as file:
    RT_rmse = float(file.read())
with open('Results/RBFNN_rmse.txt', 'r') as file:
    RBFNN_rmse = float(file.read())
with open('Results/BRNN_rmse.txt', 'r') as file:
    BRNN_rmse = float(file.read())
with open('Results/LSTM_rmse.txt', 'r') as file:
    LSTM_rmse = float(file.read())
with open('Results/GRU_rmse.txt', 'r') as file:
    GRU_rmse = float(file.read())    



# update with your real numbers
rmse_scores = {
    'SVR': SVR_rmse,
    'FFNN': FFNN_rmse,
    'KNN': KNN_rmse,
    'RT': RT_rmse,
    'RBFNN': RBFNN_rmse,
    'BRNN': BRNN_rmse,
    'LSTM': LSTM_rmse,
    'GRU': GRU_rmse
}

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(rmse_scores.keys(), rmse_scores.values(), color='skyblue')
plt.title('RMSE Comparison of Forecasting Models')
plt.ylabel('RMSE')
plt.xlabel('Model')
plt.grid(axis='y')
plt.tight_layout()
plt.show()



# RMSE Bar Chart - Well-Balanced Text Size
plt.figure(figsize=(12, 8))
sorted_rmse = dict(sorted(rmse_scores.items(), key=lambda x: x[1]))
bars = plt.bar(sorted_rmse.keys(), sorted_rmse.values(), color='slateblue', edgecolor='black', linewidth=1.2)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.4f}',
             ha='center', va='bottom', fontsize=18, fontweight='bold', color='black')

plt.title('RMSE Comparison of Forecasting Models', fontsize=30, weight='bold', pad=15)
plt.ylabel('RMSE', fontsize=24, weight='bold')
plt.xlabel('Model', fontsize=24, weight='bold')
plt.xticks(fontsize=18, weight='bold')
plt.yticks(fontsize=18, weight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("Results/Plots/RMSE_BarChart.pdf", format='pdf', dpi=600)
plt.close()

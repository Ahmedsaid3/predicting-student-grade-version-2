import pandas as pd 
import matplotlib.pyplot as plt
# rmse table 5 
data_rmse = {
    "Method": ["C-CF", "C-KM", "U-CF", "KN-CF", "Our-Model"],
    "Term-2":   [1.3615, 1.1239, 1.1602, 1.2068, 0.9543],
    "Term-3":   [1.4439, 1.0725, 1.1686, 0.9516, 0.8326],
    "Term-4":   [1.1539, 1.0396, 1.1775, 0.9802, 0.8992],
    "Term-5":   [1.1310, 1.0611, 1.2133, 0.9823, 0.9567],
    "Term-6":   [1.0885, 0.9720, 1.0276, 0.8929, 0.8853],
    "Term-7":   [1.0376, 0.9980, 1.0923, 0.9408, 1.0136],
}

a = 1
sum = 0 
print("weighted_average_rmse:")

df = pd.DataFrame(data_rmse)
kn_cf_values = df[df["Method"] == "KN-CF"].values[0][1:]  # skip the 'Method' column
for i in kn_cf_values:
    a += 1
    if (a == 2):
        sum += 9 * i
    if (a == 3):
        sum += 8 * i
    if (a == 4):
        sum += 7 * i
    if (a == 5):
        sum += 6 * i
    if (a == 6):
        sum += 8 * i
    if (a == 7):
        sum += 4 * i

weighted_average_rmse = sum / 42
print(weighted_average_rmse)

'''
df_rmse = pd.DataFrame(data_rmse)
print(df_rmse)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('off')
ax.axis('tight')

# Format numbers to 4 decimal places for display
cell_text = []
for i in range(len(df_rmse)):
    row = [df_rmse.iloc[i, 0]]  # Method column (text)
    for j in range(1, len(df_rmse.columns)):
        row.append(f"{df_rmse.iloc[i, j]:.4f}")
    cell_text.append(row)

# Create the table and add it to the axis
table = ax.table(cellText=cell_text,
                 colLabels=df_rmse.columns,
                 loc='center',
                 cellLoc='center')

# Style the table for better readability
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)  # Adjust the table size

# Highlight the minimum value in each column (starting from column 1, skipping "Method")
for j in range(1, len(df_rmse.columns)):
    col_name = df_rmse.columns[j]
    min_val = df_rmse[col_name].min()
    min_idx = df_rmse[col_name].idxmin()
    
    # Highlight the cell with the minimum value (min_idx+1 because of header row)
    cell = table.get_celld()[(min_idx+1, j)]
    cell.set_facecolor('lightgreen')  # Light green for minimum values
    cell.set_text_props(weight='bold')

plt.title('RMSE Values with Minimum Values Highlighted', fontsize=16)
plt.tight_layout()

# Save as a PNG file
plt.savefig('rmse_table_min_highlighted.png', dpi=300, bbox_inches='tight')
print("Table saved as 'rmse_table_min_highlighted.png'")

# Display the table
# plt.show()

print("\n")
'''
# mae table 6 
data_mae = {
    "Method": ["C-CF", "C-KM", "U-CF", "KN-CF", "Our-Model"],
    "Term-2":   [1.0702, 0.8464, 0.8521, 0.9280, 0.7794],
    "Term-3":   [1.1684, 0.8124, 0.8774, 0.7046, 0.6736],
    "Term-4":   [0.8435, 0.8052, 0.8545, 0.7298, 0.7303],
    "Term-5":   [0.8414, 0.8068, 0.9068, 0.7382, 0.8053],
    "Term-6":   [0.7962, 0.7238, 0.7264, 0.6418, 0.7154],
    "Term-7":   [0.7670, 0.7435, 0.8246, 0.7035, 0.8400],
}

a = 1
sum = 0 
print("weighted_average_mae:")

df = pd.DataFrame(data_mae)
kn_cf_values = df[df["Method"] == "KN-CF"].values[0][1:]  # skip the 'Method' column
for i in kn_cf_values:
    a += 1
    if (a == 2):
        sum += 9 * i
    if (a == 3):
        sum += 8 * i
    if (a == 4):
        sum += 7 * i
    if (a == 5):
        sum += 6 * i
    if (a == 6):
        sum += 8 * i
    if (a == 7):
        sum += 4 * i

weighted_average_mae = sum / 42
print(weighted_average_mae)

import json
# Your results
results = {
    "chinese educationalgraph": {
        "WeightedAvgRMSE": 1.0032285714285714,
        "WeightedAvgMAE": 0.7494047619047619
    },
}

# Save to JSON file
with open("chinese_educationalgraph.json", "w") as f:
    json.dump(results, f, indent=4)

'''
df_mae = pd.DataFrame(data_mae)
print(df_mae)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('off')
ax.axis('tight')

# Format numbers to 4 decimal places for display
cell_text = []
for i in range(len(df_mae)):
    row = [df_mae.iloc[i, 0]]  # Method column (text)
    for j in range(1, len(df_mae.columns)):
        row.append(f"{df_mae.iloc[i, j]:.4f}")
    cell_text.append(row)

# Create the table and add it to the axis
table = ax.table(cellText=cell_text,
                 colLabels=df_mae.columns,
                 loc='center',
                 cellLoc='center')

# Style the table for better readability
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)  # Adjust the table size

# Highlight the minimum value in each column (starting from column 1, skipping "Method")
for j in range(1, len(df_mae.columns)):
    col_name = df_mae.columns[j]
    min_val = df_mae[col_name].min()
    min_idx = df_mae[col_name].idxmin()
    
    # Highlight the cell with the minimum value (min_idx+1 because of header row)
    cell = table.get_celld()[(min_idx+1, j)]
    cell.set_facecolor('lightgreen')  # Light green for minimum values
    cell.set_text_props(weight='bold')

plt.title('MAE Values with Minimum Values Highlighted', fontsize=16)
plt.tight_layout()

# Save as a PNG file
plt.savefig('mae_table_min_highlighted.png', dpi=300, bbox_inches='tight')
print("Table saved as 'mae_table_min_highlighted.png'")

# Display the table
# plt.show()
'''

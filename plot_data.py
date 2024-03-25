import matplotlib.pyplot as plt
import pandas as pd

data_filename = 'bandit_results.csv'

# Load the results from a CSV file
results_df = pd.read_csv(data_filename)  # Replace with the path to your CSV file

# Plot the number of times each arm has been chosen
plt.figure(figsize=(10, 6))
plt.bar(results_df['Action'], results_df['Number of Times Chosen'], color='skyblue')
plt.xlabel('Arm')
plt.ylabel('Number of Times Chosen')
plt.title('Number of Times Each Arm Has Been Chosen')
plt.xticks(results_df['Action'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

import matplotlib.pyplot as plt
import pandas as pd

data_filename = 'bandit_results.csv'

# Load the results from a CSV file
results_df = pd.read_csv(data_filename)  # Replace with the path to your CSV file
df = pd.read_csv(data_filename)
print(df)

# Create the plot
plt.figure(figsize=(10, 6))
bars = plt.bar(df['Action'], df['Number of Times Chosen'], color='skyblue')

# Annotate bars with the estimated and true probabilities
for bar, est_prob, true_prob in zip(bars, df['Estimated Action Values'], df['True Action Values']):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'Est: {est_prob:.2f}\nTrue: {true_prob:.2f}', 
                ha='center', va='bottom', color='black', fontsize=8)

# Set the labels and title
plt.xlabel('Arm Number')
plt.ylabel('Times Chosen')
plt.title('Number of Times Each Arm Has Been Chosen')
plt.xticks(df['Action'])
plt.show()
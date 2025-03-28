import pandas as pd
import matplotlib.pyplot as plt

# Specify the path to your CSV file
csv_file_path = 'detected_objects.csv'

# Read the CSV file into a DataFrame
try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found.")
    exit()
except pd.errors.EmptyDataError:
    print("Error: The file is empty.")
    exit()

# Display the DataFrame
print(df)
print(df.describe())

# Check if the DataFrame is empty
if df.empty:
    print("No data to plot.")
else:
    # Plotting a bar graph for Object vs Count
    plt.figure(figsize=(10, 6))  # Set the figure size
    bars = plt.bar(df['Object'], df['Count'], color='skyblue')  # Create a bar graph

    # Adding title and labels
    plt.title('Detected Objects vs Count')
    plt.xlabel('Objects')
    plt.ylabel('Count')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Annotate bars with their respective counts
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

    # Adding grid lines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()  # Adjust layout to make room for labels
    plt.show()


import pandas as pd

try:
    df = pd.read_excel('jagadeesh.csv.xlsx')
    display(df.head())
    print(df.shape)
except FileNotFoundError:
    print("Error: 'jagadeesh.csv.xlsx' not found. Please ensure the file exists in the current directory or provide the correct path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
# Data Types
print("Data Types:\n", df.dtypes)

# Missing Values
print("\nMissing Values:\n", df.isnull().sum())

# Descriptive Statistics for Numerical Columns
print("\nDescriptive Statistics:\n", df.describe())

# Unique Categories and Frequencies for Categorical Columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\nUnique values in '{col}':\n{df[col].value_counts()}")

# Shape of the Dataset
print(f"\nShape of the dataset: {df.shape}")

# First Few Rows
print("\nFirst 5 rows:\n", df.head())

# Last Few Rows
print("\nLast 5 rows:\n", df.tail())

# Potential Relationships (without calculations)
print("\nPotential Relationships:")
print("The pixel values likely have a relationship with the Label.  We can investigate this further with visualization and correlation analysis.")
import matplotlib.pyplot as plt
import seaborn as sns

# Histograms for numerical features
plt.figure(figsize=(20, 15))
for i, col in enumerate(df.columns[1:65]):  # Exclude 'Image_ID' and 'Label'
    plt.subplot(8, 8, i + 1)
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(col)
plt.tight_layout()
plt.show()

# Box plots for numerical features
plt.figure(figsize=(20, 15))
for i, col in enumerate(df.columns[1:65]):  # Exclude 'Image_ID' and 'Label'
    plt.subplot(8, 8, i + 1)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# Bar chart for 'Label'
plt.figure(figsize=(10, 5))
sns.countplot(x='Label', data=df)
plt.title('Distribution of Labels')
plt.show()

# Scatter plots for selected pixels vs. 'Label'
selected_pixels = ['Pixel_1', 'Pixel_10', 'Pixel_20', 'Pixel_30', 'Pixel_40', 'Pixel_50', 'Pixel_60']
plt.figure(figsize=(15, 10))
for i, pixel in enumerate(selected_pixels):
    plt.subplot(2, 4, i + 1)
    sns.scatterplot(x=pixel, y='Label', data=df, hue='Label')
    plt.title(f'{pixel} vs. Label')
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Calculate the correlation matrix
pixel_cols = [col for col in df.columns if 'Pixel' in col]
correlation_matrix = df[pixel_cols].corr()

# Visualize the correlation matrix as a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
plt.title('Correlation Matrix of Pixel Values')
plt.show()


# Investigate the relationship between pixel values and the 'Label' column
plt.figure(figsize=(12, 6))
sns.boxplot(x='Label', y='Pixel_1', data=df) # Example, can be extended to other pixels
plt.title('Pixel_1 Values Across Different Labels')
plt.show()


# Analyze the distribution of the 'Label' column
label_counts = df['Label'].value_counts()
print(f"Distribution of Labels:\n{label_counts}")
plt.figure(figsize=(8, 6))
sns.countplot(x='Label', data=df)
plt.title('Distribution of Labels')
plt.show()


# Look for other potential patterns
# Example: Mean pixel values for each label
mean_pixel_values = df.groupby('Label')[pixel_cols].mean()
print(f"Mean pixel values by label:\n{mean_pixel_values}")

# Further analysis and observations can be added here

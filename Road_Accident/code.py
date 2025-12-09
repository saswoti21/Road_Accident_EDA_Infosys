# Import necessary libraries
import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('data/US_Accidents_March23.csv', header=0, sep = ',', nrows = 50000)

# Display the first few rows of the dataframe
print(df.head())

# Display basic information about the DataFrame, including data types and non-null values
print("\nDataFrame Info:")
df.info()

# Display descriptive statistics for numerical columns
print("\nDescriptive Statistics:")
print(df.describe())

# Drop columns that are almost entirely null: 'End_Lat', 'End_Lng', 'Wind_Chill(F)', 'Precipitation(in)'
df.drop(columns=['End_Lat', 'End_Lng', 'Wind_Chill(F)', 'Precipitation(in)'], inplace=True, errors='ignore')

# Convert 'Start_Time' and 'End_Time' to datetime objects
df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['End_Time'] = pd.to_datetime(df['End_Time'])

print("\nDataFrame Info after dropping columns and converting types:")
df.info()

# Display the number of missing values per column after initial cleanup
print("\nMissing values per column after initial cleanup:")
print(df.isnull().sum())

# Impute numerical columns with their median
numerical_cols = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)']
for col in numerical_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)

# Impute categorical columns with their mode
categorical_cols = ['Zipcode', 'Timezone', 'Airport_Code', 'Weather_Timestamp', 'Wind_Direction', 'Weather_Condition']
for col in categorical_cols:
    if df[col].isnull().any():
        mode_val = df[col].mode()[0] 
        df[col].fillna(mode_val, inplace=True)

# Verify that there are no more missing values
print("\nMissing values after imputation:")
print(df.isnull().sum().sum())

# Confirm current data types for Start_Time and End_Time
print("Current data types:")
print(df[['Start_Time', 'End_Time']].dtypes)

# Display the formatted datetime columns
print("\nFormatted Start_Time (YYYY-MM-DD HH:MM:SS):")
print(df['Start_Time'].dt.strftime('%Y-%m-%d %H:%M:%S').head())

print("\nFormatted End_Time (YYYY-MM-DD HH:MM:SS):")
print(df['End_Time'].dt.strftime('%Y-%m-%d %H:%M:%S').head())

# Extract time-based features
df['Start_Hour'] = df['Start_Time'].dt.hour
df['Start_Day_of_Week'] = df['Start_Time'].dt.day_name()
df['Start_Month'] = df['Start_Time'].dt.month_name()
df['Start_Year'] = df['Start_Time'].dt.year
df['Accident_Duration_min'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60

# Display the head of the DataFrame with new features
print(df[['Start_Time', 'Start_Hour', 'Start_Day_of_Week', 'Start_Month', 'Accident_Duration_min']].head())

# Identify categorical columns
categorical_cols_df = df.select_dtypes(include='object').columns.tolist()

# Add newly created time-based categorical features
categorical_cols_df.extend(['Start_Day_of_Week', 'Start_Month'])


print("Categorical Columns:", categorical_cols_df)

# Check the number of unique values for each categorical column
print("\nNumber of unique values per categorical column:")
for col in categorical_cols_df:
    if col in df.columns: # Check if column still exists after drops
        print(f"{col}: {df[col].nunique()} unique values")


# 1. Handle Duplicate Entries
print(f"Initial number of rows: {df.shape[0]}")
df.drop_duplicates(inplace=True)
print(f"Number of rows after removing duplicates: {df.shape[0]}")

# 2. Identify Outliers (using IQR method for numerical columns)
print("\nIdentifying outliers using IQR method...")
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    if not outliers.empty:
        print(f"\nColumn: {col}")
        print(f"  Number of outliers: {len(outliers)}")
        print(f"  Percentage of outliers: {len(outliers) / df.shape[0] * 100:.2f}%")
        print(f"  Min outlier value: {outliers[col].min()}")
        print(f"  Max outlier value: {outliers[col].max()}")
    else:
        print(f"\nColumn: {col}: No outliers detected.")

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Columns with identified outliers
outlier_cols = ['Start_Lat', 'Start_Lng', 'Visibility(mi)', 'Accident_Duration_min']

plt.figure(figsize=(15, 10))
for i, col in enumerate(outlier_cols):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f'Box Plot of {col} (Before Handling)')
    plt.ylabel(col)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Columns for which outliers were handled (same as 'before' visualization)
outlier_cols = ['Start_Lat', 'Start_Lng', 'Visibility(mi)', 'Accident_Duration_min']

plt.figure(figsize=(15, 10))
for i, col in enumerate(outlier_cols):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f'Box Plot of {col} (After handling)')
    plt.ylabel(col)
plt.tight_layout()
plt.show()
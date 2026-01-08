import milestone1 as ms1
import milestone2 as ms2

#Identify Top 5 Accident-Prone States
state_accident_counts = ms1.df['State'].value_counts()
top_5_states = state_accident_counts.head(5).index.tolist()

print("Top 5 Accident-Prone States:", top_5_states)
print("Accident Counts for Top 5 States:\n", state_accident_counts.head(5))

#Identify Top 5 Accident-Prone Cities
city_accident_counts = ms1.df['City'].value_counts()
top_5_cities = city_accident_counts.head(5).index.tolist()

print("Top 5 Accident-Prone Cities:", top_5_cities)
print("Accident Counts for Top 5 Cities:\n", city_accident_counts.head(5))

df_top_states = ms1.df[ms1.df['State'].isin(top_5_states)].copy()
df_top_cities = ms1.df[ms1.df['City'].isin(top_5_cities)].copy()

print(f"Shape of df_top_states: {df_top_states.shape}")
print(f"States in df_top_states: {df_top_states['State'].unique().tolist()}")
print(f"Shape of df_top_cities: {df_top_cities.shape}")
print(f"Cities in df_top_cities: {df_top_cities['City'].unique().tolist()}")

import matplotlib.pyplot as plt
import seaborn as sns

#Visualize Accident Hotspots in Top 5 States
plt.figure(figsize=(15, 10))
sns.scatterplot(x='Start_Lng', y='Start_Lat', hue='State', data=df_top_states, s=10, alpha=0.3, palette='viridis')
plt.title('Accident Hotspots in Top 5 States (Start Locations)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='State')
plt.tight_layout()
plt.show()

#Visualize Accident Hotspots in Top 5 Cities
plt.figure(figsize=(15, 10))
sns.scatterplot(x='Start_Lng', y='Start_Lat', hue='City', data=df_top_cities, s=10, alpha=0.3, palette='viridis')
plt.title('Accident Hotspots in Top 5 Cities (Start Locations)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='City')
plt.tight_layout()
plt.show()

#Analyze Accidents by Time of Day
plt.figure(figsize=(12, 6))
sns.countplot(x='Start_Hour', data=ms2.df_analysis, palette='viridis', hue='Start_Hour', legend=False)
plt.title('Accident Frequency by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Accidents')
plt.tight_layout()
plt.show()

rain_fog_conditions = ['Rain', 'Fog']
df_rain_fog = ms2.df_analysis[ms2.df_analysis['Weather_Condition'].isin(rain_fog_conditions)]

plt.figure(figsize=(10, 6))
sns.countplot(x='Weather_Condition', hue='Severity', data=df_rain_fog, palette='coolwarm')
plt.title('Accident Severity Distribution during Rain and Fog Conditions')
plt.xlabel('Weather Condition')
plt.ylabel('Number of Accidents')
plt.legend(title='Severity')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='Severity', y='Visibility(mi)', data=ms2.df_analysis, palette='viridis')
plt.title('Accident Severity vs. Visibility (miles)')
plt.xlabel('Severity')
plt.ylabel('Visibility (mi)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='Severity', y='Visibility(mi)', data=ms2.df_analysis, palette='viridis', hue='Severity', legend=False)
plt.title('Accident Severity vs. Visibility (miles)')
plt.xlabel('Severity')
plt.ylabel('Visibility (mi)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#Test Severity by Weather Condition
clear_weather_conditions = ['Clear', 'Fair', 'Mostly Cloudy', 'Partly Cloudy', 'Scattered Clouds', 'Cloudy']
df_clear_weather = ms2.df_analysis[ms2.df_analysis['Weather_Condition'].isin(clear_weather_conditions)]

print(f"Shape of df_rain_fog: {df_rain_fog.shape}")
print(f"Shape of df_clear_weather: {df_clear_weather.shape}")
print("First 5 rows of df_clear_weather:")
print(df_clear_weather.head())

from scipy.stats import mannwhitneyu

# Extract Severity values for rain/fog and clear weather conditions
severity_rain_fog = df_rain_fog['Severity']
severity_clear_weather = df_clear_weather['Severity']

# Perform Mann-Whitney U test
stat, p_value = mannwhitneyu(severity_rain_fog, severity_clear_weather, alternative='two-sided')

print(f"\nMann-Whitney U Test Results for Severity between Rain/Fog and Clear Weather:")
print(f"U Statistic: {stat:.2f}")
print(f"P-value: {p_value:.4f}")

# Interpret the results
alpha = 0.05
if p_value < alpha:
    print(f"Since the p-value ({p_value:.4f}) is less than alpha ({alpha}), we reject the null hypothesis.")
    print("Conclusion: There is a statistically significant difference in accident severity between rain/fog conditions and clear weather conditions.")
else:
    print(f"Since the p-value ({p_value:.4f}) is greater than alpha ({alpha}), we fail to reject the null hypothesis.")
    print("Conclusion: There is no statistically significant difference in accident severity between rain/fog conditions and clear weather conditions.")

import pandas as pd

#Test Severity by Time of Day
bins = [-1, 5, 11, 17, 23]
labels = ['Night', 'Morning', 'Afternoon', 'Evening']
ms2.df_analysis['Time_of_Day'] = pd.cut(ms2.df_analysis['Start_Hour'], bins=bins, labels=labels, ordered=False)

print("Accident counts by Time_of_Day category:")
print(ms2.df_analysis['Time_of_Day'].value_counts())
print("\nFirst 5 rows with new 'Time_of_Day' column:")
print(ms2.df_analysis[['Start_Hour', 'Time_of_Day']].head())

from scipy.stats import kruskal

# Prepare severity data for each time of day category
severity_by_time_of_day = [
    ms2.df_analysis[ms2.df_analysis['Time_of_Day'] == label]['Severity']
    for label in ms2.df_analysis['Time_of_Day'].cat.categories
]

# Perform Kruskal-Wallis H-test
stat, p_value = kruskal(*severity_by_time_of_day)

print(f"\nKruskal-Wallis H-Test Results for Severity by Time of Day:")
print(f"H-statistic: {stat:.2f}")
print(f"P-value: {p_value:.4f}")

# Interpret the results
alpha = 0.05
if p_value < alpha:
    print(f"Since the p-value ({p_value:.4f}) is less than alpha ({alpha}), we reject the null hypothesis.")
    print("Conclusion: There is a statistically significant difference in accident severity across different times of the day.")
else:
    print(f"Since the p-value ({p_value:.4f}) is greater than alpha ({alpha}), we fail to reject the null hypothesis.")
    print("Conclusion: There is no statistically significant difference in accident severity across different times of the day.")


#Test Severity by Visibility
from scipy.stats import spearmanr

# Extract Severity and Visibility(mi) columns
severity_data = ms2.df_analysis['Severity']
visibility_data = ms2.df_analysis['Visibility(mi)']

# Perform Spearman's rank correlation test
correlation_coefficient, p_value = spearmanr(severity_data, visibility_data)

print(f"\nSpearman's Rank Correlation Test Results for Severity and Visibility:")
print(f"Correlation Coefficient: {correlation_coefficient:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpret the results
alpha = 0.05
if p_value < alpha:
    print(f"Since the p-value ({p_value:.4f}) is less than alpha ({alpha}), we reject the null hypothesis.")
    print(f"Conclusion: There is a statistically significant monotonic relationship between accident severity and visibility (correlation coefficient: {correlation_coefficient:.4f}).")
else:
    print(f"Since the p-value ({p_value:.4f}) is greater than alpha ({alpha}), we fail to reject the null hypothesis.")
    print("Conclusion: There is no statistically significant monotonic relationship between accident severity and visibility.")



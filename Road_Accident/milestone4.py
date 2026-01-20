
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, kruskal, spearmanr # Import for hypothesis testing

# Set Seaborn style for better aesthetics
sns.set_style("whitegrid")

# Define road features for easy access
ROAD_FEATURES = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction',
                 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
                 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']

# 1. Load Data (and preprocessing steps done in the notebook)
@st.cache_data
def load_data():
    # Ensure the path to your CSV file is correct
    df = pd.read_csv('data/US_Accidents_March23.csv', header=0, sep = ',', nrows = 700000)

    # Preprocessing (from notebook)
    df.drop(columns=['End_Lat', 'End_Lng', 'Wind_Chill(F)', 'Precipitation(in)'], inplace=True, errors='ignore')
    df['Start_Time'] = pd.to_datetime(df['Start_Time'])
    df['End_Time'] = pd.to_datetime(df['End_Time'])

    numerical_cols = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)']
    for col in numerical_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val) # Changed from inplace=True

    categorical_cols = ['Zipcode', 'Timezone', 'Airport_Code', 'Weather_Timestamp', 'Wind_Direction', 'Weather_Condition']
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0] # .mode() can return multiple values, pick the first
            df[col] = df[col].fillna(mode_val) # Changed from inplace=True

    # Fill remaining NaNs for 'Street', 'City', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight'
    for col in ['Street', 'City', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight']:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0]) # Changed from inplace=True

    # Extract time-based features (after initial fillna, as they rely on Start_Time)
    # Moved this calculation *before* outlier capping
    df['Start_Hour'] = df['Start_Time'].dt.hour
    df['Start_Day_of_Week'] = df['Start_Time'].dt.day_name()
    df['Start_Month'] = df['Start_Time'].dt.month_name()
    df['Accident_Duration_min'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60

    # Capping outliers (as done in notebook)
    outlier_cols_to_cap = ['Start_Lat', 'Start_Lng', 'Distance(mi)', 'Temperature(F)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Accident_Duration_min']
    for col in outlier_cols_to_cap:
        lower_bound = df[col].quantile(0.05)
        upper_bound = df[col].quantile(0.95)
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    # Categorize Start_Hour into Time_of_Day for hypothesis testing
    bins = [-1, 5, 11, 17, 23]
    labels = ['Night', 'Morning', 'Afternoon', 'Evening']
    df['Time_of_Day'] = pd.cut(df['Start_Hour'], bins=bins, labels=labels, ordered=False)

    return df

df = load_data()

# 2. Streamlit App Layout
st.set_page_config(layout="wide", page_title="US Accidents Analysis")
st.title("US Accidents Data Analysis Dashboard")

# --- Sidebar for Navigation and Filters ---
st.sidebar.title("Navigation")
analysis_selection = st.sidebar.radio(
    "Go to",
    ["Overview", "Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis", "Geospatial Analysis", "Hypothesis Testing", "Help"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Global Filters")
st.sidebar.markdown("Adjust the filters below to refine your analysis across all sections.")

# Global Filters
selected_severity = st.sidebar.multiselect(
    "Select Severity Level(s)",
    options=sorted(df['Severity'].unique().tolist()),
    default=sorted(df['Severity'].unique().tolist()),
    help="Filter accidents by severity level (1-4)."
)

available_states = df['State'].dropna().unique().tolist()
selected_states = st.sidebar.multiselect(
    "Select State(s)",
    options=sorted(available_states),
    default=sorted(available_states),
    help="Filter accidents by state."
)

min_visibility, max_visibility = float(df['Visibility(mi)'].min()), float(df['Visibility(mi)'].max())
selected_visibility_range = st.sidebar.slider(
    "Select Visibility Range (mi)",
    min_value=min_visibility,
    max_value=max_visibility,
    value=(min_visibility, max_visibility),
    help="Filter accidents by visibility in miles."
)

selected_road_features = st.sidebar.multiselect(
    "Filter by Road Conditions",
    options=ROAD_FEATURES,
    help="Filter accidents where AT LEAST ONE of the selected road features is present."
)

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
selected_days = st.sidebar.multiselect(
    "Select Day(s) of Week",
    options=day_order,
    default=day_order,
    help="Filter accidents by day of the week."
)

month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
selected_months = st.sidebar.multiselect(
    "Select Month(s)",
    options=month_order,
    default=month_order,
    help="Filter accidents by month."
)

min_hour, max_hour = int(df['Start_Hour'].min()), int(df['Start_Hour'].max())
selected_hour_range = st.sidebar.slider(
    "Select Hour Range (24-hour clock)",
    min_value=min_hour,
    max_value=max_hour,
    value=(min_hour, max_hour),
    help="Filter accidents by hour of the day."
)

# Apply global filters
filtered_df = df[
    (df['Severity'].isin(selected_severity)) &
    (df['State'].isin(selected_states)) &
    (df['Visibility(mi)'] >= selected_visibility_range[0]) &
    (df['Visibility(mi)'] <= selected_visibility_range[1]) &
    (df['Start_Day_of_Week'].isin(selected_days)) &
    (df['Start_Month'].isin(selected_months)) &
    (df['Start_Hour'] >= selected_hour_range[0]) &
    (df['Start_Hour'] <= selected_hour_range[1])
].copy() # Use .copy() to avoid SettingWithCopyWarning

# Apply road feature filter
if selected_road_features:
    # Filter where at least one of the selected road features is True
    road_feature_mask = filtered_df[selected_road_features].any(axis=1)
    filtered_df = filtered_df[road_feature_mask]

# Caching for frequently used filtered data derived from filtered_df
@st.cache_data
def get_top_n_counts(df_source, column, n=10):
    return df_source[column].value_counts().head(n)

@st.cache_data
def get_plot(df_source, plot_type, x_col, y_col=None, hue_col=None, order=None, title="", xlabel="", ylabel="", palette='viridis', legend=True, xtick_rotation=0):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Fundamental check: if the DataFrame itself is empty, don't attempt to plot
    if df_source.empty:
        ax.text(0.5, 0.5, "No data to display with current filters.", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(title)
        ax.axis('off')
        plt.tight_layout() # Added for consistency
        return fig # Return the figure with the "No data" message

    can_plot = True
    # Check if x_col is valid and has data
    if x_col not in df_source.columns or df_source[x_col].isnull().all() or df_source[x_col].nunique() == 0:
        can_plot = False
    # Check if y_col is valid and has data (if provided)
    if y_col and (y_col not in df_source.columns or df_source[y_col].isnull().all() or df_source[y_col].nunique() == 0):
        can_plot = False
    # Check if hue_col is valid and has data (if provided)
    if hue_col and (hue_col not in df_source.columns or df_source[hue_col].isnull().all()):
        # For hue, it's okay if nunique is 1, but not 0 or all nulls
        if df_source[hue_col].nunique() == 0:
            can_plot = False


    if not can_plot:
        ax.text(0.5, 0.5, f"Insufficient data for {x_col} {'and ' + y_col if y_col else ''} to plot.", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(title)
        ax.axis('off')
        plt.tight_layout() # Added for consistency
        return fig

    try:
        if plot_type == 'countplot':
            sns.countplot(x=x_col, data=df_source, hue=hue_col, order=order, palette=palette, legend=legend, ax=ax)
        elif plot_type == 'barplot':
            sns.barplot(x=x_col, y=y_col, data=df_source, hue=hue_col, order=order, palette=palette, legend=legend, ax=ax)
        elif plot_type == 'scatterplot':
            sns.scatterplot(x=x_col, y=y_col, hue=hue_col, data=df_source, s=10, alpha=0.3, palette=palette, ax=ax)
        elif plot_type == 'violinplot':
            sns.violinplot(x=x_col, y=y_col, data=df_source, hue=hue_col, palette=palette, legend=legend, ax=ax)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xtick_rotation:
            ax.tick_params(axis='x', rotation=xtick_rotation, ha='right')
        plt.tight_layout()
    except Exception as e:
        ax.text(0.5, 0.5, f"Error plotting: {e}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(title)
        ax.axis('off')
        plt.tight_layout()

    return fig


if filtered_df.empty:
    st.warning("No data matches the selected filters. Please adjust your selections.")
else:
    if analysis_selection == "Overview":
        st.header("Overview: US Accidents Data")
        st.write("This section provides a high-level summary of the accident data based on the applied filters.")
        st.metric(label="Total Accidents (Filtered)", value=len(filtered_df))

        st.markdown("---")
        st.subheader("Top Accident-Prone States")
        top_states = get_top_n_counts(filtered_df, 'State', n=5)
        if not top_states.empty: # Added check
            st.bar_chart(top_states)
            st.write(top_states)
        else:
            st.info("No data for Top Accident-Prone States with current filters.")

        st.subheader("Top Accident-Prone Cities")
        top_cities = get_top_n_counts(filtered_df, 'City', n=5)
        if not top_cities.empty: # Added check
            st.bar_chart(top_cities)
            st.write(top_cities)
        else:
            st.info("No data for Top Accident-Prone Cities with current filters.")

        st.markdown("---")
        st.subheader("Accident Severity Distribution")
        fig_severity = get_plot(filtered_df, 'countplot', 'Severity', hue_col='Severity', legend=False, title='Distribution of Accident Severity', xlabel='Severity', ylabel='Number of Accidents')
        st.pyplot(fig_severity)
        plt.close(fig_severity)

        st.subheader("Accident Frequency by Hour of Day")
        fig_hour_overview = get_plot(filtered_df, 'countplot', 'Start_Hour', hue_col='Start_Hour', legend=False, title='Accident Frequency by Hour of Day', xlabel='Hour of Day', ylabel='Number of Accidents')
        st.pyplot(fig_hour_overview)
        plt.close(fig_hour_overview)

    elif analysis_selection == "Univariate Analysis":
        st.header("Univariate Analysis: Individual Feature Distributions")
        st.write("Explore the distribution of individual features for the filtered accident data.")

        st.subheader("Accident Frequency by Day of Week")
        fig_day_week = get_plot(filtered_df, 'countplot', 'Start_Day_of_Week', hue_col='Start_Day_of_Week', order=day_order, palette='magma', legend=False, title='Accident Frequency by Day of Week', xlabel='Day of Week', ylabel='Number of Accidents', xtick_rotation=45)
        st.pyplot(fig_day_week)
        plt.close(fig_day_week)

        st.subheader("Accident Frequency by Month")
        fig_month = get_plot(filtered_df, 'countplot', 'Start_Month', hue_col='Start_Month', order=month_order, palette='viridis', legend=False, title='Accident Frequency by Month', xlabel='Month', ylabel='Number of Accidents', xtick_rotation=45)
        st.pyplot(fig_month)
        plt.close(fig_month)

        st.subheader("Accident Frequency by Weather Condition")
        weather_counts_order = filtered_df['Weather_Condition'].value_counts().head(10).index.tolist()
        fig_weather = get_plot(filtered_df, 'countplot', 'Weather_Condition', hue_col='Weather_Condition', order=weather_counts_order, palette='coolwarm', legend=False, title='Top 10 Weather Conditions during Accidents', xlabel='Weather Condition', ylabel='Number of Accidents', xtick_rotation=45)
        st.pyplot(fig_weather)
        plt.close(fig_weather)

        st.subheader("Accident Frequency by Sunrise/Sunset")
        fig_sunrise = get_plot(filtered_df, 'countplot', 'Sunrise_Sunset', hue_col='Sunrise_Sunset', palette='plasma', legend=False, title='Accident Frequency by Time of Day (Sunrise/Sunset)', xlabel='Time of Day', ylabel='Number of Accidents')
        st.pyplot(fig_sunrise)
        plt.close(fig_sunrise)

        st.subheader("Frequency of Road Features at Accident Locations")
        road_feature_counts_series = filtered_df[ROAD_FEATURES].sum().sort_values(ascending=False)
        road_feature_counts_df = road_feature_counts_series.reset_index(name='Count') # Renamed to 'Count'
        road_feature_counts_df.columns = ['Road Feature', 'Count'] # Ensure column names are correct for barplot

        if not road_feature_counts_df.empty and road_feature_counts_df['Count'].sum() > 0: # Check if there's any 'True' count
            fig_road_features = get_plot(road_feature_counts_df, 'barplot', 'Road Feature', 'Count', palette='coolwarm', legend=False, title='Frequency of Road Features at Accident Locations', xlabel='Road Feature', ylabel='Number of Accidents (Feature Present)', xtick_rotation=45)
            st.pyplot(fig_road_features)
            plt.close(fig_road_features)
        else:
            st.info("No road feature data for selected filters.")


    elif analysis_selection == "Bivariate Analysis":
        st.header("Bivariate Analysis: Relationships Between Variables")
        st.write("Examine relationships between two variables to uncover potential insights.")

        st.subheader("Accident Severity Distribution Across Top Weather Conditions")
        top_10_weather_conditions = filtered_df['Weather_Condition'].value_counts().head(10).index.tolist()
        df_top_weather_bivar = filtered_df[filtered_df['Weather_Condition'].isin(top_10_weather_conditions)].copy()
        fig_weather_severity = get_plot(df_top_weather_bivar, 'countplot', 'Weather_Condition', hue_col='Severity', palette='viridis', title='Accident Severity Distribution Across Top 10 Weather Conditions', xlabel='Weather Condition', ylabel='Number of Accidents', xtick_rotation=45)
        st.pyplot(fig_weather_severity)
        plt.close(fig_weather_severity)


        st.subheader("Accident Severity vs. Visibility (miles)")
        fig_visibility_severity = get_plot(filtered_df, 'violinplot', 'Severity', 'Visibility(mi)', hue_col='Severity', legend=False, palette='viridis', title='Accident Severity vs. Visibility (miles)', xlabel='Severity', ylabel='Visibility (mi)')
        st.pyplot(fig_visibility_severity)
        plt.close(fig_visibility_severity)


        st.subheader("Accident Severity Distribution by Hour of Day")
        fig_hour_severity = get_plot(filtered_df, 'countplot', 'Start_Hour', hue_col='Severity', palette='viridis', order=sorted(filtered_df['Start_Hour'].unique()), title='Accident Severity Distribution by Hour of Day', xlabel='Hour of Day', ylabel='Number of Accidents')
        st.pyplot(fig_hour_severity)
        plt.close(fig_hour_severity)

        st.subheader("Accident Severity Distribution by Road Features")
        st.markdown("Hover over the plots to see the distribution of severity when a specific road feature is present.")

        plots_to_show = []
        for feature in ROAD_FEATURES:
            df_feature_present = filtered_df[filtered_df[feature] == True]
            if not df_feature_present.empty and df_feature_present['Severity'].nunique() > 0:
                plots_to_show.append((feature, df_feature_present))

        if plots_to_show:
            num_cols = 3
            num_rows = (len(plots_to_show) + num_cols - 1) // num_cols
            fig_road_features_severity, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
            axes = axes.flatten() # Flatten for easy iteration

            for idx, (feature, df_fp) in enumerate(plots_to_show):
                ax = axes[idx]
                sns.countplot(x='Severity', hue='Severity', data=df_fp, palette='viridis', legend=False, ax=ax)
                ax.set_title(f'Severity Dist. when {feature.replace("_", " ")} is Present')
                ax.set_xlabel('Severity')
                ax.set_ylabel('Number of Accidents')

            # Hide any unused subplots
            for j in range(len(plots_to_show), len(axes)):
                axes[j].axis('off')

            plt.tight_layout()
            st.pyplot(fig_road_features_severity)
            plt.close(fig_road_features_severity)
        else:
            st.info("No road feature data with severity information for selected filters.")


    elif analysis_selection == "Multivariate Analysis":
        st.header("Multivariate Analysis: Correlations")
        st.write("Explore relationships among multiple numerical variables, including their correlation with accident severity.")

        correlation_columns = [
            'Severity',
            'Visibility(mi)',
            'Accident_Duration_min',
            'Temperature(F)',
            'Humidity(%)',
            'Pressure(in)',
            'Wind_Speed(mph)'
        ]
        valid_correlation_cols = [col for col in correlation_columns if col in filtered_df.columns and filtered_df[col].dtype in ['float64', 'int64']]
        if len(valid_correlation_cols) < 2:
            st.warning("Not enough numerical data to compute correlation matrix with current filters.")
        else:
            correlation_matrix = filtered_df[valid_correlation_cols].corr()
            if not correlation_matrix.empty: # Added check
                st.subheader("Correlation Heatmap")
                fig_heatmap, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
                ax.set_title('Correlation Heatmap of Numerical Variables and Severity')
                st.pyplot(fig_heatmap)
                plt.close(fig_heatmap)
            else:
                st.info("No valid correlations to display with current filters.")

            st.subheader("Pair Plots of Selected Numerical Variables by Severity")
            pair_plot_cols = ['Severity', 'Visibility(mi)', 'Distance(mi)', 'Temperature(F)']
            valid_pair_plot_cols = [col for col in pair_plot_cols if col in filtered_df.columns]
            # Ensure there's enough data for pairplot and at least 2 numeric columns if no hue, or 1 numeric + hue
            if len(valid_pair_plot_cols) > 1 and not filtered_df[valid_pair_plot_cols].empty:
                df_pair_plot = filtered_df[valid_pair_plot_cols].copy()
                # Check for sufficient unique values for hue and numeric cols for plotting
                has_sufficient_hue = 'Severity' in df_pair_plot.columns and df_pair_plot['Severity'].nunique() > 1
                has_sufficient_numeric_cols = len([c for c in valid_pair_plot_cols if c != 'Severity' and df_pair_plot[c].nunique() > 1]) >= 1

                if has_sufficient_hue or (not has_sufficient_hue and has_sufficient_numeric_cols):
                    try:
                        if has_sufficient_hue:
                            pair_plot_fig = sns.pairplot(df_pair_plot, hue='Severity', palette='viridis', diag_kind='kde')
                        else:
                            pair_plot_fig = sns.pairplot(df_pair_plot, palette='viridis', diag_kind='kde') # No hue if only one severity
                        pair_plot_fig.fig.suptitle('Pair Plots of Selected Numerical Variables by Severity', y=1.02)
                        st.pyplot(pair_plot_fig)
                        plt.close(pair_plot_fig) # Close the figure after displaying
                    except Exception as e:
                        st.error(f"Error generating pair plot: {e}")
                else:
                    st.info("Not enough diverse data for pair plots with current filters.")
            else:
                st.info("Not enough data or columns for pair plots with current filters.")

    elif analysis_selection == "Geospatial Analysis":
        st.header("Geospatial Analysis: Accident Hotspots")
        st.write("Visualize geographical concentrations of accidents in selected states and cities.")

        st.subheader("Accident Hotspots in Selected States")
        fig_state_hotspots = get_plot(filtered_df, 'scatterplot', 'Start_Lng', 'Start_Lat', hue_col='State', palette='viridis', title='Accident Hotspots in Selected States (Start Locations)', xlabel='Longitude', ylabel='Latitude')
        st.pyplot(fig_state_hotspots)
        plt.close(fig_state_hotspots)

        st.subheader("Accident Hotspots in Selected Cities")
        fig_city_hotspots = get_plot(filtered_df, 'scatterplot', 'Start_Lng', 'Start_Lat', hue_col='City', palette='viridis', title='Accident Hotspots in Selected Cities (Start Locations)', xlabel='Longitude', ylabel='Latitude')
        st.pyplot(fig_city_hotspots)
        plt.close(fig_city_hotspots)

    elif analysis_selection == "Hypothesis Testing":
        st.header("Hypothesis Testing Results")
        st.write("This section presents the results of statistical tests performed to evaluate specific hypotheses about accident severity.")

        alpha = 0.05

        # --- Hypothesis 1: Severity vs. Time of Day ---
        st.subheader("1. Accident Severity and Time of Day (Kruskal-Wallis H-Test)")
        st.markdown("""
        *   **Null Hypothesis (H0)**: There is no significant relationship between accident severity and the time of day (hour of occurrence).
        *   **Alternative Hypothesis (H1)**: There is a significant relationship between accident severity and the time of day (hour of occurrence). Accidents tend to be more or less severe during certain hours.
        """)

        if 'Time_of_Day' in filtered_df.columns and not filtered_df['Time_of_Day'].empty and filtered_df['Severity'].nunique() > 0:
            severity_by_time_of_day = [
                filtered_df[filtered_df['Time_of_Day'] == label]['Severity']
                for label in filtered_df['Time_of_Day'].cat.categories
                if not filtered_df[filtered_df['Time_of_Day'] == label]['Severity'].empty
            ]
            severity_by_time_of_day = [s for s in severity_by_time_of_day if not s.empty]

            if len(severity_by_time_of_day) > 1: # Kruskal requires at least 2 groups
                stat_time_of_day, p_value_time_of_day = kruskal(*severity_by_time_of_day)
                st.write(f"**H-statistic:** {stat_time_of_day:.2f}")
                st.write(f"**P-value:** {p_value_time_of_day:.4f}")
                if p_value_time_of_day < alpha:
                    st.success(f"Conclusion: Reject H0. There is a statistically significant difference in accident severity across different times of the day (p-value: {p_value_time_of_day:.4f}).")
                else:
                    st.info(f"Conclusion: Fail to reject H0. There is no statistically significant difference in accident severity across different times of the day (p-value: {p_value_time_of_day:.4f}).")
            else:
                st.warning("Not enough time of day categories with data to perform Kruskal-Wallis test with current filters.")
        else:
            st.warning("Time of Day or Severity data is not available with current filters for hypothesis testing.")

        st.markdown("---")
        # --- Hypothesis 2: Severity vs. Rain/Fog Conditions ---
        st.subheader("2. Accident Severity and Rain/Fog Conditions (Mann-Whitney U Test)")
        st.markdown("""
        *   **Null Hypothesis (H0)**: There is no significant difference in accident severity between rain/fog conditions and clear weather conditions.
        *   **Alternative Hypothesis (H1)**: Accidents are significantly more severe during rain or fog conditions compared to clear weather conditions.
        """)

        rain_fog_conditions = ['Rain', 'Fog', 'Heavy Rain', 'Light Rain', 'Rain Showers']
        clear_weather_conditions = ['Clear', 'Fair', 'Mostly Cloudy', 'Partly Cloudy', 'Scattered Clouds', 'Cloudy']

        severity_rain_fog = filtered_df[filtered_df['Weather_Condition'].isin(rain_fog_conditions)]['Severity']
        severity_clear_weather = filtered_df[filtered_df['Weather_Condition'].isin(clear_weather_conditions)]['Severity']

        if not severity_rain_fog.empty and not severity_clear_weather.empty:
            stat_weather, p_value_weather = mannwhitneyu(severity_rain_fog, severity_clear_weather, alternative='two-sided')
            st.write(f"**U Statistic:** {stat_weather:.2f}")
            st.write(f"**P-value:** {p_value_weather:.4f}")
            if p_value_weather < alpha:
                st.success(f"Conclusion: Reject H0. There is a statistically significant difference in accident severity between rain/fog and clear weather conditions (p-value: {p_value_weather:.4f}).")
            else:
                st.info(f"Conclusion: Fail to reject H0. There is no statistically significant difference in accident severity between rain/fog and clear weather conditions (p-value: {p_value_weather:.4f}).")
        else:
            st.warning("Not enough data for both rain/fog and clear weather conditions to perform Mann-Whitney U test with current filters.")

        st.markdown("---")
        # --- Hypothesis 3: Severity vs. Visibility ---
        st.subheader("3. Accident Severity and Visibility (Spearman's Rank Correlation)")
        st.markdown("""
        *   **Null Hypothesis (H0)**: There is no significant correlation between accident severity and visibility.
        *   **Alternative Hypothesis (H1)**: There is a significant inverse correlation between accident severity and visibility. Lower visibility conditions are associated with higher accident severity.
        """)

        if not filtered_df['Severity'].empty and not filtered_df['Visibility(mi)'].empty:
            correlation_coefficient, p_value_visibility = spearmanr(filtered_df['Severity'], filtered_df['Visibility(mi)'])
            st.write(f"**Correlation Coefficient:** {correlation_coefficient:.4f}")
            st.write(f"**P-value:** {p_value_visibility:.4f}")
            if p_value_visibility < alpha:
                st.success(f"Conclusion: Reject H0. There is a statistically significant monotonic relationship between accident severity and visibility (correlation coefficient: {correlation_coefficient:.4f}, p-value: {p_value_visibility:.4f}).")
            else:
                st.info(f"Conclusion: Fail to reject H0. There is no statistically significant monotonic relationship between accident severity and visibility (p-value: {p_value_visibility:.4f}).")
        else:
            st.warning("Not enough data for Severity or Visibility to perform Spearman's Rank Correlation with current filters.")

    elif analysis_selection == "Help":
        st.header("Help & Information")
        st.write("Welcome to the US Accidents Data Analysis Dashboard!")
        st.markdown("""
        This interactive dashboard allows you to explore and analyze accident data from various perspectives.
        Use the sidebar filters to refine the data displayed across all analysis tabs.

        ### How to Use the Dashboard

        1.  **Navigation**: Use the radio buttons in the sidebar under "Navigation" to switch between different analysis sections.
        2.  **Global Filters**:
            *   **Select Severity Level(s)**: Choose one or more accident severity levels (1-4) to include in the analysis.
            *   **Select State(s)**: Filter accidents by specific US states.
            *   **Select Visibility Range (mi)**: Adjust the slider to focus on accidents within a particular visibility range.
            *   **Filter by Road Conditions**: Select road features (e.g., Junction, Traffic Signal) to see accidents where at least one of these features is present.
            *   **Select Day(s) of Week**: Filter accidents by specific days of the week.
            *   **Select Month(s)**: Filter accidents by months of the year.
            *   **Select Hour Range**: Narrow down accidents to specific hours of the day using the slider.
            *   _Note_: If "No data matches the selected filters" appears, try adjusting your selections.

        ### Tab Descriptions

        *   **Overview**: Provides a summary of total accidents, top accident-prone states/cities, overall severity distribution, and accident frequency by hour.
        *   **Univariate Analysis**: Explores the distributions of individual features like day of week, month, weather conditions, sunrise/sunset, and road features.
        *   **Bivariate Analysis**: Visualizes relationships between two variables, such as severity vs. weather condition, severity vs. visibility, severity vs. hour, and severity with various road features.
        *   **Multivariate Analysis**: Displays correlations between multiple numerical variables using a heatmap and pair plots for selected variables.
        *   **Geospatial Analysis**: Shows accident hotspots by plotting accident locations (latitude and longitude) for selected states and cities.
        *   **Hypothesis Testing**: Presents the formal statistical tests performed to evaluate specific hypotheses regarding accident severity and its relationship with time of day, weather conditions, and visibility.
        *   **Help**: This page, providing guidance on how to use the dashboard and understand its content.

        ### Data Source
        The data used for this analysis is a subset of the US Accidents dataset (March 2023), obtained from Kaggle.
        """)




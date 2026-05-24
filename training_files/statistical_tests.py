import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", context="talk", palette="viridis")
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 20,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

SEVERITY_PALETTE = {1: '#d9534f', 2: '#f0ad4e', 3: '#5cb85c'}
SEVERITY_LABELS = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}

def check_dir(path):
    """Ensures directory exists."""
    os.makedirs(path, exist_ok=True)

def generate_descriptive_summary(df):
    """
    Generates a descriptive summary table of all variables in the dataset.
    Reports: Data Type, Non-Null Count, Null Percentage, Unique Count, 
             Mean, Median, Std Dev, Min, Max, and Skewness.
    """
    summary_data = []
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].count()
        null_pct = (df[col].isnull().sum() / len(df)) * 100
        unique_cnt = df[col].nunique()
        
        # Calculate numeric statistics if applicable
        if np.issubdtype(df[col].dtype, np.number):
            mean_val = f"{df[col].mean():.3f}"
            median_val = f"{df[col].median():.3f}"
            std_val = f"{df[col].std():.3f}"
            min_val = f"{df[col].min():.3f}"
            max_val = f"{df[col].max():.3f}"
            skew_val = f"{df[col].skew():.3f}"
        else:
            mean_val = "N/A"
            median_val = "N/A"
            std_val = "N/A"
            min_val = "N/A"
            max_val = "N/A"
            skew_val = "N/A"
            
        summary_data.append({
            "Column Name": col,
            "Data Type": dtype,
            "Non-Null Count": non_null,
            "Null Percentage (%)": f"{null_pct:.2f}%",
            "Unique Values": unique_cnt,
            "Mean": mean_val,
            "Median": median_val,
            "Std Dev": std_val,
            "Min": min_val,
            "Max": max_val,
            "Skewness": skew_val
        })
        
    return pd.DataFrame(summary_data)

def plot_univariate_distributions(df, output_dir='reports/figures'):
    """
    Plots the distributions of key variables:
    1. Target Variable (accident_severity)
    2. Speed Limit
    3. Time of Day
    4. Season
    5. Accident Hour
    """
    check_dir(output_dir)
    plots_saved = {}
    
    # 1. Target Variable Distribution
    plt.figure(figsize=(10, 6))
    severity_counts = df['accident_severity'].value_counts().sort_index()
    total = len(df)
    
    # Map numerical categories to clean text labels
    labels = [f"{SEVERITY_LABELS[i]} (Class {i})" for i in severity_counts.index]
    
    ax = sns.barplot(
        x=labels, 
        y=severity_counts.values, 
        palette=[SEVERITY_PALETTE[i] for i in severity_counts.index],
        edgecolor='black',
        linewidth=1.2
    )
    
    for p in ax.patches:
        height = p.get_height()
        pct = (height / total) * 100
        ax.annotate(f'{height:,}\n({pct:.2f}%)',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=12, fontweight='bold', xytext=(0, 5),
                    textcoords='offset points')
                    
    plt.title("Distribution of Accident Severity (Target Variable)", fontsize=18, fontweight='bold', pad=15)
    plt.ylabel("Number of Accidents", fontsize=14)
    plt.xlabel("Severity Level", fontsize=14)
    plt.ylim(0, max(severity_counts.values) * 1.15)
    plt.tight_layout()
    target_path = os.path.join(output_dir, 'univariate_target.png')
    plt.savefig(target_path)
    plt.close()
    plots_saved['target'] = target_path

    plt.figure(figsize=(10, 6))
    speed_counts = df['speed_limit'].value_counts().sort_index()
    ax = sns.barplot(
        x=speed_counts.index.astype(int), 
        y=speed_counts.values, 
        palette="viridis",
        edgecolor='black',
        linewidth=1
    )
    for p in ax.patches:
        height = p.get_height()
        pct = (height / total) * 100
        ax.annotate(f'{pct:.1f}%',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10, xytext=(0, 3),
                    textcoords='offset points')
    plt.title("Distribution of Speed Limits in Accidents", fontsize=18, fontweight='bold', pad=15)
    plt.ylabel("Number of Accidents", fontsize=14)
    plt.xlabel("Speed Limit (mph)", fontsize=14)
    plt.ylim(0, max(speed_counts.values) * 1.1)
    plt.tight_layout()
    speed_path = os.path.join(output_dir, 'univariate_speed_limit.png')
    plt.savefig(speed_path)
    plt.close()
    plots_saved['speed_limit'] = speed_path

    if 'time_of_day' in df.columns:
        plt.figure(figsize=(10, 6))
        tod_counts = df['time_of_day'].value_counts()
        ax = sns.barplot(
            x=tod_counts.index, 
            y=tod_counts.values, 
            palette="magma",
            edgecolor='black',
            linewidth=1
        )
        for p in ax.patches:
            height = p.get_height()
            pct = (height / total) * 100
            ax.annotate(f'{pct:.1f}%',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=11, xytext=(0, 3),
                        textcoords='offset points')
        plt.title("Distribution of Accidents by Time of Day", fontsize=18, fontweight='bold', pad=15)
        plt.ylabel("Number of Accidents", fontsize=14)
        plt.xlabel("Time of Day", fontsize=14)
        plt.ylim(0, max(tod_counts.values) * 1.1)
        plt.tight_layout()
        tod_path = os.path.join(output_dir, 'univariate_time_of_day.png')
        plt.savefig(tod_path)
        plt.close()
        plots_saved['time_of_day'] = tod_path

    plt.figure(figsize=(12, 6))
    sns.histplot(
        data=df, 
        x='accident_hour', 
        bins=24, 
        kde=True, 
        color='#4b6b94', 
        edgecolor='black', 
        alpha=0.8
    )
    plt.title("Hourly Distribution of Accidents", fontsize=18, fontweight='bold', pad=15)
    plt.xlabel("Hour of the Day (0-23)", fontsize=14)
    plt.ylabel("Number of Accidents", fontsize=14)
    plt.xticks(range(0, 24))
    plt.xlim(-0.5, 23.5)
    plt.tight_layout()
    hour_path = os.path.join(output_dir, 'univariate_accident_hour.png')
    plt.savefig(hour_path)
    plt.close()
    plots_saved['accident_hour'] = hour_path

    return plots_saved

def plot_bivariate_analysis(df_orig, output_dir='reports/figures'):
    """
    Plots stacked bar charts showing the proportion of accident severity across different categories:
    1. Urban vs. Rural Area
    2. Speed Limit
    3. Weather Conditions
    4. Road Surface Conditions
    5. Light Conditions
    6. Time of Day
    """
    check_dir(output_dir)
    df = df_orig.copy()
    plots_saved = {}
    
    # Helper to plot 100% stacked bar chart
    def plot_stacked_pct(feature, title, xlabel, filename):
        # Create cross tabulation
        ct = pd.crosstab(df[feature], df['accident_severity'])
        # Reorder columns to ensure: Fatal (1), Serious (2), Slight (3)
        ct = ct.reindex(columns=[1, 2, 3], fill_value=0)
        # Convert to row percentages
        ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
        
        fig, ax = plt.subplots(figsize=(11, 7))
        
        ct_pct.plot(
            kind='bar', 
            stacked=True, 
            ax=ax, 
            color=[SEVERITY_PALETTE[i] for i in [1, 2, 3]],
            edgecolor='black', 
            width=0.7
        )
        
        for container in ax.containers:
            labels = [f'{val:.1f}%' if val > 2 else '' for val in container.datavalues]
            ax.bar_label(container, labels=labels, label_type='center', fontsize=10, color='white', fontweight='bold')
            
        plt.title(title, fontsize=18, fontweight='bold', pad=15)
        plt.ylabel("Percentage of Accidents (%)", fontsize=14)
        plt.xlabel(xlabel, fontsize=14)
        plt.xticks(rotation=0 if df[feature].nunique() <= 5 else 45, ha='right' if df[feature].nunique() > 5 else 'center')
        
        handles, leg_labels = ax.get_legend_handles_labels()
        ax.legend(handles, ['Fatal (Class 1)', 'Serious (Class 2)', 'Slight (Class 3)'], 
                  title="Severity", bbox_to_anchor=(1.02, 1), loc='upper left')
                  
        plt.tight_layout()
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        return save_path

    df['urban_or_rural_area'] = df['urban_or_rural_area'].map({1: 'Urban', 2: 'Rural'}).fillna('Unknown')
    plots_saved['urban_rural'] = plot_stacked_pct(
        'urban_or_rural_area', 
        "Accident Severity Proportions: Urban vs. Rural Areas", 
        "Area Category", 
        'bivariate_urban_rural.png'
    )
    
    plots_saved['speed_limit'] = plot_stacked_pct(
        'speed_limit', 
        "Accident Severity Proportions across Speed Limits", 
        "Speed Limit (mph)", 
        'bivariate_speed_limit.png'
    )
    
    weather_mapping = {
        1: 'Fine (No Wind)', 
        2: 'Rain (No Wind)', 
        3: 'Snow (No Wind)',
        4: 'Fine + Wind', 
        5: 'Rain + Wind', 
        6: 'Snow + Wind',
        7: 'Fog/Mist', 
        8: 'Other', 
        9: 'Unknown'
    }
    df['weather_conditions_text'] = df['weather_conditions'].map(weather_mapping).fillna('Other/Unknown')
    plots_saved['weather'] = plot_stacked_pct(
        'weather_conditions_text', 
        "Accident Severity Proportions across Weather Conditions", 
        "Weather Condition", 
        'bivariate_weather.png'
    )

    light_mapping = {
        1: 'Daylight',
        4: 'Dark (Lights Lit)',
        5: 'Dark (Lights Unlit)',
        6: 'Dark (No Lights)',
        7: 'Dark (Lights Unknown)'
    }
    df['light_conditions_text'] = df['light_conditions'].map(light_mapping).fillna('Other/Unknown')
    plots_saved['light'] = plot_stacked_pct(
        'light_conditions_text', 
        "Accident Severity Proportions across Light Conditions", 
        "Light Condition", 
        'bivariate_light.png'
    )

    if 'time_of_day' in df.columns:
        plots_saved['time_of_day'] = plot_stacked_pct(
            'time_of_day', 
            "Accident Severity Proportions across Times of Day", 
            "Time of Day", 
            'bivariate_time_of_day.png'
        )

    return plots_saved

def run_inferential_tests(df):
    """
    Performs rigorous statistical hypothesis testing:
    1. Chi-Square Test of Independence (with Cramer's V effect size) for all categorical variables.
    2. Kruskal-Wallis H-Test for numeric variables (speed_limit, accident_hour) across target groups.
    """
    categorical_features = [
        'urban_or_rural_area', 'speed_limit', 'light_conditions', 'weather_conditions', 
        'road_surface_conditions', 'junction_detail', 'is_weekend', 'time_of_day', 'season'
    ]
    
    categorical_features = [col for col in categorical_features if col in df.columns]
    
    chi_results = []
    
    for feature in categorical_features:
        contingency_table = pd.crosstab(df[feature], df['accident_severity'])
        
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
        
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0
        
        if cramers_v < 0.07:
            interpretation = "Negligible / Very Weak"
        elif cramers_v < 0.21:
            interpretation = "Weak"
        elif cramers_v < 0.35:
            interpretation = "Moderate"
        else:
            interpretation = "Strong"
            
        chi_results.append({
            "Feature": feature,
            "Chi-Square Stat": f"{chi2:.3f}",
            "Degrees of Freedom": dof,
            "p-value": f"{p_val:.3e}" if p_val < 0.001 else f"{p_val:.4f}",
            "Cramer's V": f"{cramers_v:.4f}",
            "Association Strength": interpretation
        })
        
    chi_df = pd.DataFrame(chi_results).sort_values(by="Cramer's V", ascending=False)
    
    kw_features = ['speed_limit', 'accident_hour']
    kw_results = []
    
    for feature in kw_features:
        if feature in df.columns:
            group1 = df[df['accident_severity'] == 1][feature].values
            group2 = df[df['accident_severity'] == 2][feature].values
            group3 = df[df['accident_severity'] == 3][feature].values
            
            h_stat, p_val = stats.kruskal(group1, group2, group3)
            
            kw_results.append({
                "Numeric Feature": feature,
                "H-Statistic": f"{h_stat:.3f}",
                "p-value": f"{p_val:.3e}" if p_val < 0.001 else f"{p_val:.4f}",
                "Statistically Significant": "Yes (p < 0.001)" if p_val < 0.001 else "No"
            })
            
    kw_df = pd.DataFrame(kw_results)
    
    return chi_df, kw_df

def plot_correlation_heatmap(df, output_dir='reports/figures'):
    check_dir(output_dir)
    
    cols_to_drop = ['accident_datetime', 'time_of_day', 'season', 'accident_seriousness']
    df_numeric = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    df_numeric = df_numeric.select_dtypes(include=[np.number])
    
    var = df_numeric.var()
    df_numeric = df_numeric.loc[:, var > 0]
    
    correlations = df_numeric.corr()['accident_severity'].abs().sort_values(ascending=False)
    top_corr_features = correlations.head(15).index.tolist()
    
    corr_matrix = df_numeric[top_corr_features].corr()
    
    plt.figure(figsize=(14, 12))
    
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        fmt='.2f', 
        linewidths=0.5, 
        vmin=-1, 
        vmax=1,
        annot_kws={"size": 10, "weight": "bold"}
    )
    
    plt.title("Correlation Matrix Heatmap of Top 15 Predictive Features", fontsize=18, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'multivariate_correlation_heatmap.png')
    plt.savefig(save_path)
    plt.close()
    
    return save_path

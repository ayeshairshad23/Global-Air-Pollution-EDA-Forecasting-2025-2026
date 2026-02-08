"""
================================================================================
GLOBAL AIR POLLUTION ANALYSIS
================================================================================
A comprehensive data analysis script for analyzing global air pollution patterns.

Author: Data Analyst
Date: 2025-2026
Purpose: Load, clean, analyze, and visualize air pollution data with insights
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from scipy import stats

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set display options for better output formatting
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Configure visualization styles
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ================================================================================
# 1. DATA LOADING AND EXPLORATION
# ================================================================================

def load_dataset(file_path):
    """
    Load air pollution dataset with encoding handling.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING AND EXPLORATION")
    print("="*80)
    
    try:
        # Try UTF-8 encoding first
        df = pd.read_csv(file_path, encoding='utf-8')
        print("✓ Dataset loaded successfully with UTF-8 encoding")
    except UnicodeDecodeError:
        # Fallback to other encodings
        try:
            df = pd.read_csv(file_path, encoding='latin-1')
            print("✓ Dataset loaded successfully with latin-1 encoding")
        except:
            df = pd.read_csv(file_path, encoding='iso-8859-1')
            print("✓ Dataset loaded successfully with iso-8859-1 encoding")
    
    return df


def display_basic_structure(df):
    """Display basic dataset structure and information."""
    
    print("\n" + "-"*80)
    print("DATASET SHAPE AND BASIC INFO")
    print("-"*80)
    print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print("\n📋 Column Names and Data Types:")
    print(df.dtypes)
    
    print("\n📊 First 5 Rows:")
    print(df.head())
    
    print("\n📊 Last 5 Rows:")
    print(df.tail())
    
    print("\n📋 Dataset Info:")
    df.info()


# ================================================================================
# 2. DATA UNDERSTANDING
# ================================================================================

def understand_data(df):
    """
    Perform comprehensive data understanding analysis.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    print("\n" + "="*80)
    print("STEP 2: DATA UNDERSTANDING")
    print("="*80)
    
    print("\n" + "-"*80)
    print("STATISTICAL SUMMARY")
    print("-"*80)
    print(df.describe().round(3))
    
    print("\n" + "-"*80)
    print("UNIQUE VALUES IN CATEGORICAL COLUMNS")
    print("-"*80)
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        unique_count = df[col].nunique()
        print(f"\n{col}: {unique_count} unique values")
        if unique_count <= 15:
            print(df[col].value_counts())
        else:
            print(f"(Too many unique values to display - showing top 10)")
            print(df[col].value_counts().head(10))
    
    print("\n" + "-"*80)
    print("DATA TYPES SUMMARY")
    print("-"*80)
    print(f"Numeric columns: {df.select_dtypes(include=[np.number]).shape[1]}")
    print(f"Categorical columns: {df.select_dtypes(include=['object']).shape[1]}")
    print(f"Datetime columns: {df.select_dtypes(include=['datetime64']).shape[1]}")


# ================================================================================
# 3. DATA CLEANING
# ================================================================================

def clean_data(df):
    """
    Perform comprehensive data cleaning including:
    - Missing value handling
    - Duplicate removal
    - Outlier treatment
    - Column name standardization
    - Date conversion
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print("\n" + "="*80)
    print("STEP 3: DATA CLEANING")
    print("="*80)
    
    df_clean = df.copy()
    
    # --- Missing Values Analysis ---
    print("\n" + "-"*80)
    print("MISSING VALUES ANALYSIS")
    print("-"*80)
    
    missing_data = pd.DataFrame({
        'Column': df_clean.columns,
        'Missing_Count': df_clean.isnull().sum(),
        'Missing_Percentage': (df_clean.isnull().sum() / len(df_clean) * 100).round(2)
    })
    missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    if len(missing_data) == 0:
        print("✓ No missing values detected!")
    else:
        print(missing_data.to_string(index=False))
        
        # Handle missing values
        print("\n📌 Handling missing values...")
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                # Use median for numeric columns (robust to outliers)
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                print(f"  ✓ {col}: Filled with median value ({median_val:.2f})")
        
        # For categorical columns, use mode or 'Unknown'
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna('Unknown', inplace=True)
                print(f"  ✓ {col}: Filled with 'Unknown'")
    
    # --- Duplicate Removal ---
    print("\n" + "-"*80)
    print("DUPLICATE RECORDS CHECK")
    print("-"*80)
    
    duplicate_count = df_clean.duplicated().sum()
    if duplicate_count == 0:
        print("✓ No duplicate records found!")
    else:
        print(f"⚠ Found {duplicate_count} duplicate records")
        df_clean = df_clean.drop_duplicates()
        print(f"✓ Removed {duplicate_count} duplicates. New shape: {df_clean.shape}")
    
    # --- Column Name Standardization ---
    print("\n" + "-"*80)
    print("STANDARDIZING COLUMN NAMES")
    print("-"*80)
    
    # Convert to lowercase and replace spaces with underscores
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_').str.replace('.', '')
    print("✓ Columns standardized to lowercase with underscores")
    print(f"New columns: {list(df_clean.columns)}")
    
    # --- Date Conversion ---
    print("\n" + "-"*80)
    print("DATE/TIME CONVERSION")
    print("-"*80)
    
    if 'date' in df_clean.columns:
        df_clean['date'] = pd.to_datetime(df_clean['date'])
        print("✓ 'date' column converted to datetime")
    
    # --- Outlier Detection and Treatment ---
    print("\n" + "-"*80)
    print("OUTLIER DETECTION AND TREATMENT (IQR METHOD)")
    print("-"*80)
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    outliers_summary = {}
    
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_count = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
        if outlier_count > 0:
            outliers_summary[col] = {
                'count': outlier_count,
                'percentage': (outlier_count / len(df_clean) * 100),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
    
    if outliers_summary:
        print("\nOutliers detected:")
        for col, info in outliers_summary.items():
            print(f"  {col}: {info['count']} outliers ({info['percentage']:.2f}%)")
            print(f"    Bounds: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]")
        
        # Treat outliers by clipping (keeping them but capping extreme values)
        print("\n📌 Applying IQR-based clipping to outliers...")
        for col in outliers_summary.keys():
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
        print("✓ Outliers clipped to IQR bounds")
    else:
        print("✓ No significant outliers detected!")
    
    print("\n" + "-"*80)
    print("CLEANING SUMMARY")
    print("-"*80)
    print(f"✓ Cleaned dataset shape: {df_clean.shape}")
    print(f"✓ Rows removed: {df.shape[0] - df_clean.shape[0]}")
    
    return df_clean


# ================================================================================
# 4. FEATURE ENGINEERING
# ================================================================================

def engineer_features(df):
    """
    Create new features from existing data:
    - Extract temporal features (year, month, day, hour, season)
    - Create pollution index
    - Create severity categories
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        
    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
    print("\n" + "="*80)
    print("STEP 4: FEATURE ENGINEERING")
    print("="*80)
    
    df_eng = df.copy()
    
    # --- Temporal Features ---
    if 'date' in df_eng.columns:
        print("\n📌 Extracting temporal features...")
        df_eng['year'] = df_eng['date'].dt.year
        df_eng['month'] = df_eng['date'].dt.month
        df_eng['day'] = df_eng['date'].dt.day
        df_eng['hour'] = df_eng['date'].dt.hour
        df_eng['dayofweek'] = df_eng['date'].dt.dayofweek
        df_eng['dayofweek_name'] = df_eng['date'].dt.day_name()
        df_eng['quarter'] = df_eng['date'].dt.quarter
        
        # Create season feature
        season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                     3: 'Spring', 4: 'Spring', 5: 'Spring',
                     6: 'Summer', 7: 'Summer', 8: 'Summer',
                     9: 'Fall', 10: 'Fall', 11: 'Fall'}
        df_eng['season'] = df_eng['month'].map(season_map)
        
        print("✓ Extracted: year, month, day, hour, dayofweek, quarter, season")
    
    # --- Pollution Index Creation ---
    print("\n📌 Creating pollution metrics...")
    
    # Define pollutant columns
    pollutant_cols = ['pm25', 'pm10', 'no2', 'so2', 'co', 'ozone']
    available_pollutants = [col for col in pollutant_cols if col in df_eng.columns]
    
    if available_pollutants:
        # Normalize and create composite pollution index
        df_normalized = df_eng[available_pollutants].copy()
        
        for col in available_pollutants:
            max_val = df_normalized[col].max()
            min_val = df_normalized[col].min()
            if max_val > min_val:
                df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        
        # Weighted pollution index (PM2.5 and PM10 have higher weights)
        weights = {
            'pm25': 0.35,
            'pm10': 0.25,
            'no2': 0.15,
            'so2': 0.10,
            'co': 0.10,
            'ozone': 0.05
        }
        
        df_eng['pollution_index'] = 0
        for col in available_pollutants:
            df_eng['pollution_index'] += df_normalized[col] * weights.get(col, 1/len(available_pollutants))
        
        df_eng['pollution_index'] = df_eng['pollution_index'] * 100  # Scale to 0-100
        print(f"✓ Created 'pollution_index' from {len(available_pollutants)} pollutants")
    
    # --- Severity Category ---
    if 'aqi_class' in df_eng.columns:
        print("✓ Using existing 'aqi_class' for severity levels")
    elif 'pollution_index' in df_eng.columns:
        def categorize_severity(score):
            if score < 20:
                return 'Good'
            elif score < 40:
                return 'Moderate'
            elif score < 60:
                return 'Unhealthy for Sensitive'
            elif score < 80:
                return 'Unhealthy'
            else:
                return 'Hazardous'
        
        df_eng['pollution_severity'] = df_eng['pollution_index'].apply(categorize_severity)
        print("✓ Created 'pollution_severity' based on pollution_index")
    
    # --- Extract Location Information ---
    if 'city' in df_eng.columns:
        # Extract country from city column if present
        df_eng['location'] = df_eng['city'].str.strip()
        print("✓ Standardized location information")
    
    print("\n" + "-"*80)
    print("ENGINEERED FEATURES SUMMARY")
    print("-"*80)
    print(f"New columns created: {set(df_eng.columns) - set(df.columns)}")
    print(f"Total columns now: {df_eng.shape[1]}")
    
    return df_eng


# ================================================================================
# 5. EXPLORATORY DATA ANALYSIS (EDA)
# ================================================================================

def eda_distributions(df):
    """Analyze distributions of major pollutants."""
    
    print("\n" + "="*80)
    print("STEP 5: EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*80)
    
    print("\n" + "-"*80)
    print("5.1 DISTRIBUTION ANALYSIS OF POLLUTANTS")
    print("-"*80)
    
    pollutant_cols = ['pm25', 'pm10', 'no2', 'so2', 'co', 'ozone']
    available_pollutants = [col for col in pollutant_cols if col in df.columns]
    
    # Statistical summary for pollutants
    print("\n📊 Pollutant Statistics:")
    pollutant_stats = df[available_pollutants].describe().round(2)
    print(pollutant_stats)
    
    # Visualization: Distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Distribution of Major Pollutants', fontsize=16, fontweight='bold')
    
    axes = axes.ravel()
    for idx, col in enumerate(available_pollutants):
        axes[idx].hist(df[col], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{col.upper()} Distribution', fontweight='bold')
        axes[idx].set_xlabel(f'{col.upper()} Level')
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('01_pollutant_distributions.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: 01_pollutant_distributions.png")
    plt.show()
    
    # Insight
    print("\n💡 INSIGHTS FROM DISTRIBUTION ANALYSIS:")
    print("-" * 80)
    for col in available_pollutants:
        mean_val = df[col].mean()
        median_val = df[col].median()
        skewness = df[col].skew()
        print(f"{col.upper()}:")
        print(f"  Mean: {mean_val:.2f}, Median: {median_val:.2f}, Skewness: {skewness:.2f}")
        if skewness > 0.5:
            print(f"  → Right-skewed distribution (higher tail)")
        elif skewness < -0.5:
            print(f"  → Left-skewed distribution (lower tail)")
        else:
            print(f"  → Fairly symmetric distribution")


def eda_correlation(df):
    """Analyze correlation between pollutants."""
    
    print("\n" + "-"*80)
    print("5.2 CORRELATION ANALYSIS")
    print("-"*80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    print("\nCorrelation with PM2.5 (strongest indicator of air pollution):")
    if 'pm25' in correlation_matrix.columns:
        correlations = correlation_matrix['pm25'].sort_values(ascending=False)
        print(correlations.round(3))
    
    # Visualization: Correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, mask=mask, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Pollutants and Atmospheric Conditions', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('02_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: 02_correlation_heatmap.png")
    plt.show()
    
    # Insight
    print("\n💡 INSIGHTS FROM CORRELATION ANALYSIS:")
    print("-" * 80)
    if 'pm25' in correlation_matrix.columns:
        strong_corr = correlation_matrix['pm25'][correlation_matrix['pm25'].abs() > 0.5]
        print(f"Pollutants strongly correlated with PM2.5:")
        for pollutant, corr_value in strong_corr.items():
            if pollutant != 'pm25':
                print(f"  {pollutant.upper()}: {corr_value:.3f}")


def eda_city_comparison(df):
    """Compare pollution levels across cities."""
    
    print("\n" + "-"*80)
    print("5.3 CITY-WISE POLLUTION COMPARISON")
    print("-"*80)
    
    if 'city' not in df.columns:
        print("City information not available in dataset")
        return
    
    # Get pollution averages by city
    pollutant_cols = ['pm25', 'pm10', 'no2', 'so2', 'co', 'ozone', 'pollution_index']
    available_cols = [col for col in pollutant_cols if col in df.columns]
    
    city_pollution = df.groupby('city')[available_cols].mean().round(2)
    city_pollution = city_pollution.sort_values('pm25', ascending=False)
    
    print("\n📊 Average Pollution by City (Top 10):")
    print(city_pollution.head(10))
    
    # Visualization: Top 10 most polluted cities
    if len(city_pollution) > 0:
        top_cities = city_pollution.head(10).reset_index()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # PM2.5 by city
        axes[0].barh(range(len(top_cities)), top_cities['pm25'], color='coral')
        axes[0].set_yticks(range(len(top_cities)))
        axes[0].set_yticklabels(top_cities['city'], fontsize=9)
        axes[0].set_xlabel('PM2.5 Level', fontweight='bold')
        axes[0].set_title('Top 10 Most Polluted Cities (PM2.5)', fontweight='bold')
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # PM10 by city
        axes[1].barh(range(len(top_cities)), top_cities['pm10'], color='lightcoral')
        axes[1].set_yticks(range(len(top_cities)))
        axes[1].set_yticklabels(top_cities['city'], fontsize=9)
        axes[1].set_xlabel('PM10 Level', fontweight='bold')
        axes[1].set_title('Top 10 Most Polluted Cities (PM10)', fontweight='bold')
        axes[1].invert_yaxis()
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('03_city_pollution_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: 03_city_pollution_comparison.png")
        plt.show()
    
    # Insight
    print("\n💡 INSIGHTS FROM CITY COMPARISON:")
    print("-" * 80)
    worst_city = city_pollution.index[0]
    best_city = city_pollution.index[-1]
    print(f"Most polluted city: {worst_city}")
    print(f"Least polluted city: {best_city}")
    print(f"Difference in PM2.5: {city_pollution['pm25'].iloc[0] - city_pollution['pm25'].iloc[-1]:.2f}")


def eda_time_trends(df):
    """Analyze pollution trends over time."""
    
    print("\n" + "-"*80)
    print("5.4 TIME TREND ANALYSIS")
    print("-"*80)
    
    if 'date' not in df.columns:
        print("Date information not available")
        return
    
    # Group by date and calculate mean pollution
    daily_pollution = df.groupby(df['date'].dt.date)[['pm25', 'pm10', 'no2', 'pollution_index']].mean()
    
    print("\n📊 Daily Pollution Trend (First 10 days):")
    print(daily_pollution.head(10).round(2))
    
    # Visualization: Time series
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # PM2.5 and PM10 trend
    axes[0].plot(daily_pollution.index, daily_pollution['pm25'], label='PM2.5', marker='o', markersize=3)
    axes[0].plot(daily_pollution.index, daily_pollution['pm10'], label='PM10', marker='s', markersize=3)
    axes[0].set_ylabel('Concentration Level', fontweight='bold')
    axes[0].set_title('PM2.5 and PM10 Trends Over Time', fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Pollution Index trend
    axes[1].plot(daily_pollution.index, daily_pollution['pollution_index'], 
                color='darkred', marker='o', markersize=3, linewidth=2)
    axes[1].fill_between(range(len(daily_pollution)), daily_pollution['pollution_index'], alpha=0.3, color='red')
    axes[1].set_xlabel('Date', fontweight='bold')
    axes[1].set_ylabel('Pollution Index', fontweight='bold')
    axes[1].set_title('Overall Pollution Index Trend', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('04_time_trends.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: 04_time_trends.png")
    plt.show()
    
    # Insight
    print("\n💡 INSIGHTS FROM TIME TREND ANALYSIS:")
    print("-" * 80)
    pm25_trend = daily_pollution['pm25'].iloc[-1] - daily_pollution['pm25'].iloc[0]
    print(f"PM2.5 trend: {pm25_trend:+.2f} (from first to last day)")
    if pm25_trend > 0:
        print(f"  → Air quality WORSENING over time")
    elif pm25_trend < 0:
        print(f"  → Air quality IMPROVING over time")
    else:
        print(f"  → Relatively stable air quality")


def eda_hourly_patterns(df):
    """Analyze hourly pollution patterns."""
    
    print("\n" + "-"*80)
    print("5.5 HOURLY POLLUTION PATTERNS")
    print("-"*80)
    
    if 'hour' not in df.columns:
        print("Hour information not available")
        return
    
    # Group by hour
    hourly_pollution = df.groupby('hour')[['pm25', 'pm10', 'no2', 'ozone']].mean()
    
    print("\n📊 Average Hourly Pollution Levels:")
    print(hourly_pollution.round(2))
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Hourly Pollution Patterns (24-hour cycle)', fontsize=14, fontweight='bold')
    
    pollutants = ['pm25', 'pm10', 'no2', 'ozone']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    axes = axes.ravel()
    for idx, pollutant in enumerate(pollutants):
        if pollutant in hourly_pollution.columns:
            axes[idx].plot(hourly_pollution.index, hourly_pollution[pollutant], 
                          marker='o', linewidth=2, markersize=6, color=colors[idx])
            axes[idx].fill_between(hourly_pollution.index, hourly_pollution[pollutant], 
                                   alpha=0.3, color=colors[idx])
            axes[idx].set_xlabel('Hour of Day', fontweight='bold')
            axes[idx].set_ylabel(f'{pollutant.upper()} Level', fontweight='bold')
            axes[idx].set_title(f'{pollutant.upper()} - Hourly Pattern', fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    plt.savefig('05_hourly_patterns.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: 05_hourly_patterns.png")
    plt.show()
    
    # Insight
    print("\n💡 INSIGHTS FROM HOURLY PATTERNS:")
    print("-" * 80)
    if 'pm25' in hourly_pollution.columns:
        peak_hour = hourly_pollution['pm25'].idxmax()
        low_hour = hourly_pollution['pm25'].idxmin()
        print(f"Peak PM2.5 pollution: {peak_hour}:00 hours")
        print(f"Lowest PM2.5 pollution: {low_hour}:00 hours")
        print(f"Difference: {hourly_pollution['pm25'].max() - hourly_pollution['pm25'].min():.2f} units")
        print(f"  → Suggests {peak_hour}:00 is typically the most polluted time")


def eda_seasonal_patterns(df):
    """Analyze seasonal pollution patterns."""
    
    print("\n" + "-"*80)
    print("5.6 SEASONAL POLLUTION PATTERNS")
    print("-"*80)
    
    if 'season' not in df.columns:
        print("Season information not available")
        return
    
    # Group by season
    seasonal_pollution = df.groupby('season')[['pm25', 'pm10', 'no2', 'pollution_index']].agg(['mean', 'std', 'min', 'max'])
    
    print("\n📊 Seasonal Pollution Statistics:")
    print(seasonal_pollution.round(2))
    
    # Visualization
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    seasonal_means = df.groupby('season')[['pm25', 'pm10', 'no2']].mean().reindex(season_order)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(season_order))
    width = 0.25
    
    ax.bar(x - width, seasonal_means['pm25'], width, label='PM2.5', color='#FF6B6B')
    ax.bar(x, seasonal_means['pm10'], width, label='PM10', color='#4ECDC4')
    ax.bar(x + width, seasonal_means['no2'], width, label='NO2', color='#45B7D1')
    
    ax.set_xlabel('Season', fontweight='bold')
    ax.set_ylabel('Average Pollution Level', fontweight='bold')
    ax.set_title('Seasonal Pollution Patterns', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(season_order)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('06_seasonal_patterns.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: 06_seasonal_patterns.png")
    plt.show()
    
    # Insight
    print("\n💡 INSIGHTS FROM SEASONAL ANALYSIS:")
    print("-" * 80)
    worst_season = seasonal_means['pm25'].idxmax()
    best_season = seasonal_means['pm25'].idxmin()
    print(f"Most polluted season: {worst_season}")
    print(f"Least polluted season: {best_season}")
    print(f"Seasonal variation in PM2.5: {seasonal_means['pm25'].max() - seasonal_means['pm25'].min():.2f} units")


def eda_aqi_distribution(df):
    """Analyze AQI class distribution."""
    
    print("\n" + "-"*80)
    print("5.7 AIR QUALITY INDEX (AQI) DISTRIBUTION")
    print("-"*80)
    
    if 'aqi_class' not in df.columns:
        print("AQI class information not available")
        return
    
    aqi_counts = df['aqi_class'].value_counts()
    aqi_percentage = (aqi_counts / len(df) * 100).round(2)
    
    print("\n📊 AQI Class Distribution:")
    aqi_summary = pd.DataFrame({
        'Count': aqi_counts,
        'Percentage': aqi_percentage
    })
    print(aqi_summary)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart
    colors_aqi = {'Good': '#2ecc71', 'Moderate': '#f39c12', 'Unhealthy for Sensitive': '#e74c3c', 
                  'Unhealthy': '#c0392b', 'Hazardous': '#8b0000'}
    pie_colors = [colors_aqi.get(cat, '#95a5a6') for cat in aqi_counts.index]
    
    axes[0].pie(aqi_counts, labels=aqi_counts.index, autopct='%1.1f%%', 
               colors=pie_colors, startangle=90)
    axes[0].set_title('Distribution of Air Quality Classes', fontweight='bold')
    
    # Bar chart
    axes[1].bar(range(len(aqi_counts)), aqi_counts.values, color=pie_colors)
    axes[1].set_xticks(range(len(aqi_counts)))
    axes[1].set_xticklabels(aqi_counts.index, rotation=45, ha='right')
    axes[1].set_ylabel('Count', fontweight='bold')
    axes[1].set_title('AQI Class Frequency', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('07_aqi_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: 07_aqi_distribution.png")
    plt.show()
    
    # Insight
    print("\n💡 INSIGHTS FROM AQI ANALYSIS:")
    print("-" * 80)
    most_common = aqi_counts.index[0]
    print(f"Most common AQI class: {most_common} ({aqi_percentage.iloc[0]}%)")
    
    unhealthy_pct = aqi_percentage[aqi_counts.index.isin(['Unhealthy', 'Unhealthy for Sensitive', 'Hazardous'])].sum()
    print(f"Percentage of unhealthy days: {unhealthy_pct:.2f}%")


# ================================================================================
# 6. MAIN ANALYSIS PIPELINE
# ================================================================================

def main():
    """Execute complete analysis pipeline."""
    
    print("\n" + "="*80)
    print("GLOBAL AIR POLLUTION DATA ANALYSIS")
    print("="*80)
    print(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 1. Load data
    file_path = 'Global_Air_Pollution_Data_2025_2026.csv'
    df = load_dataset(file_path)
    display_basic_structure(df)
    
    # 2. Understand data
    understand_data(df)
    
    # 3. Clean data
    df_clean = clean_data(df)
    
    # 4. Engineer features
    df_eng = engineer_features(df_clean)
    
    # 5. EDA
    print("\n" + "="*80)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*80)
    
    eda_distributions(df_eng)
    eda_correlation(df_eng)
    eda_city_comparison(df_eng)
    eda_time_trends(df_eng)
    eda_hourly_patterns(df_eng)
    eda_seasonal_patterns(df_eng)
    eda_aqi_distribution(df_eng)
    
    # 6. Final Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\n📊 Generated Visualizations:")
    print("  ✓ 01_pollutant_distributions.png")
    print("  ✓ 02_correlation_heatmap.png")
    print("  ✓ 03_city_pollution_comparison.png")
    print("  ✓ 04_time_trends.png")
    print("  ✓ 05_hourly_patterns.png")
    print("  ✓ 06_seasonal_patterns.png")
    print("  ✓ 07_aqi_distribution.png")
    
    print("\n📌 Key Recommendations:")
    print("  1. Focus mitigation efforts on peak pollution hours (typically early morning/evening)")
    print("  2. Prepare seasonal interventions for winter months (typically most polluted)")
    print("  3. Monitor cities with consistently high PM2.5 levels")
    print("  4. Implement stricter regulations during high pollution periods")
    print("  5. Encourage public awareness campaigns in high AQI areas")
    
    print("\n" + "="*80)
    
    return df_eng


if __name__ == "__main__":
    # Execute analysis
    df_final = main()
    
    # Save cleaned data for future use
    df_final.to_csv('Global_Air_Pollution_Cleaned_Analyzed.csv', index=False)
    print("\n✓ Saved cleaned dataset: Global_Air_Pollution_Cleaned_Analyzed.csv")

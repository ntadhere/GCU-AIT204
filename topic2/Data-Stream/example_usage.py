#!/usr/bin/env python3
"""
Example Usage of Collected Fraud Data
======================================
This script demonstrates how to load and analyze the fraud detection dataset
collected by fraud_collector.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def find_latest_dataset():
    """Find the most recently collected dataset"""
    data_files = list(Path('.').glob('fraud_data_*.csv'))
    if not data_files:
        print("❌ No datasets found!")
        print("   Run: python3 fraud_collector.py --size 2000")
        return None
    
    latest = max(data_files, key=lambda p: p.stat().st_mtime)
    print(f"📁 Found dataset: {latest}")
    return latest


def load_and_explore(filename):
    """Load dataset and show basic exploration"""
    print("\n" + "=" * 70)
    print("LOADING AND EXPLORING DATASET")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv(filename)
    
    print(f"\n📊 Dataset Shape: {df.shape}")
    print(f"   Rows (transactions): {len(df)}")
    print(f"   Columns (features): {len(df.columns)}")
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    print(f"\n🔍 Missing values: {missing}")
    
    # Class distribution
    print(f"\n📈 Class Distribution:")
    fraud_counts = df['is_fraud'].value_counts()
    fraud_rate = df['is_fraud'].mean()
    print(f"   Normal (0): {fraud_counts[0]} ({(1-fraud_rate)*100:.2f}%)")
    print(f"   Fraud (1):  {fraud_counts[1]} ({fraud_rate*100:.2f}%)")
    
    # Basic statistics
    print(f"\n💰 Transaction Amounts:")
    print(f"   Mean:   ${df['amount'].mean():.2f}")
    print(f"   Median: ${df['amount'].median():.2f}")
    print(f"   Std:    ${df['amount'].std():.2f}")
    print(f"   Min:    ${df['amount'].min():.2f}")
    print(f"   Max:    ${df['amount'].max():.2f}")
    
    # Feature types
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n🔢 Numeric features: {len(numeric_features)}")
    
    return df


def analyze_fraud_patterns(df):
    """Analyze fraud patterns in the dataset"""
    print("\n" + "=" * 70)
    print("FRAUD PATTERN ANALYSIS")
    print("=" * 70)
    
    fraud_df = df[df['is_fraud'] == 1]
    
    if 'fraud_pattern' in df.columns:
        print("\n🚨 Fraud Patterns Detected:")
        patterns = fraud_df['fraud_pattern'].value_counts()
        for pattern, count in patterns.items():
            pct = (count / len(fraud_df)) * 100
            print(f"   {pattern:20s}: {count:3d} ({pct:5.1f}%)")
    
    # User profile analysis
    if 'user_profile' in df.columns:
        print("\n👤 Fraud by User Profile:")
        profile_fraud = df.groupby('user_profile')['is_fraud'].agg(['sum', 'count', 'mean'])
        profile_fraud = profile_fraud.sort_values('mean', ascending=False).head(10)
        print("\n   Top 10 profiles by fraud rate:")
        for profile, row in profile_fraud.iterrows():
            print(f"   {profile:20s}: {row['sum']:3.0f}/{row['count']:3.0f} ({row['mean']*100:5.1f}%)")


def analyze_feature_importance(df):
    """Analyze which features correlate with fraud"""
    print("\n" + "=" * 70)
    print("FEATURE CORRELATION ANALYSIS")
    print("=" * 70)
    
    # Select numeric features (exclude metadata)
    exclude_cols = ['transaction_id', 'user_id', 'timestamp', 'is_fraud']
    feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in exclude_cols]
    
    # Calculate correlations with fraud label
    correlations = df[feature_cols + ['is_fraud']].corr()['is_fraud'].drop('is_fraud')
    correlations = correlations.abs().sort_values(ascending=False)
    
    print("\n🎯 Top 15 Features Correlated with Fraud:")
    print("\n   Feature                          Correlation")
    print("   " + "-" * 48)
    for feature, corr in correlations.head(15).items():
        bar = "█" * int(corr * 50)
        print(f"   {feature:30s}: {corr:.4f} {bar}")
    
    return correlations


def visualize_data(df):
    """Create visualizations of the fraud data"""
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Fraud Detection Dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. Amount distribution by fraud status
    ax1 = plt.subplot(3, 3, 1)
    df[df['is_fraud'] == 0]['amount'].hist(bins=30, alpha=0.7, label='Normal', ax=ax1)
    ax1.set_xlabel('Amount ($)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Normal Transaction Amounts')
    ax1.legend()
    
    ax2 = plt.subplot(3, 3, 2)
    df[df['is_fraud'] == 1]['amount'].hist(bins=30, alpha=0.7, label='Fraud', 
                                           color='red', ax=ax2)
    ax2.set_xlabel('Amount ($)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Fraudulent Transaction Amounts')
    ax2.legend()
    
    # 2. Fraud rate by hour
    ax3 = plt.subplot(3, 3, 3)
    fraud_by_hour = df.groupby('hour_of_day')['is_fraud'].mean() * 100
    ax3.bar(fraud_by_hour.index, fraud_by_hour.values, color='coral')
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Fraud Rate (%)')
    ax3.set_title('Fraud Rate by Hour of Day')
    ax3.grid(True, alpha=0.3)
    
    # 3. Amount z-score distribution
    ax4 = plt.subplot(3, 3, 4)
    df[df['is_fraud'] == 0]['amount_zscore'].hist(bins=30, alpha=0.7, 
                                                   label='Normal', ax=ax4)
    df[df['is_fraud'] == 1]['amount_zscore'].hist(bins=30, alpha=0.7, 
                                                   label='Fraud', color='red', ax=ax4)
    ax4.set_xlabel('Amount Z-Score')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Amount Z-Score Distribution')
    ax4.legend()
    ax4.axvline(x=3, color='black', linestyle='--', label='Z=3')
    
    # 4. User transaction count
    ax5 = plt.subplot(3, 3, 5)
    normal_counts = df[df['is_fraud'] == 0]['user_transaction_count']
    fraud_counts = df[df['is_fraud'] == 1]['user_transaction_count']
    ax5.boxplot([normal_counts, fraud_counts], labels=['Normal', 'Fraud'])
    ax5.set_ylabel('Transaction Count')
    ax5.set_title('User Transaction Count Distribution')
    ax5.grid(True, alpha=0.3)
    
    # 5. Fraud by weekend
    ax6 = plt.subplot(3, 3, 6)
    weekend_fraud = df.groupby('is_weekend')['is_fraud'].mean() * 100
    ax6.bar(['Weekday', 'Weekend'], weekend_fraud.values, color=['skyblue', 'orange'])
    ax6.set_ylabel('Fraud Rate (%)')
    ax6.set_title('Fraud Rate: Weekday vs Weekend')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 6. Amount deviation scatter
    ax7 = plt.subplot(3, 3, 7)
    normal = df[df['is_fraud'] == 0]
    fraud = df[df['is_fraud'] == 1]
    ax7.scatter(normal['amount'], normal['amount_deviation'], 
               alpha=0.3, s=10, label='Normal')
    ax7.scatter(fraud['amount'], fraud['amount_deviation'], 
               alpha=0.5, s=20, color='red', label='Fraud')
    ax7.set_xlabel('Amount ($)')
    ax7.set_ylabel('Amount Deviation')
    ax7.set_title('Amount vs Deviation from User Average')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 7. Time pattern heatmap
    ax8 = plt.subplot(3, 3, 8)
    time_features = ['is_night', 'is_business_hours', 'is_lunch_time', 'is_evening']
    if all(col in df.columns for col in time_features):
        time_fraud = df.groupby(time_features)['is_fraud'].mean().reset_index()
        pivot = time_fraud.pivot_table(index='is_night', columns='is_business_hours', 
                                       values='is_fraud', aggfunc='mean')
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='Reds', ax=ax8)
        ax8.set_title('Fraud Rate Heatmap: Night vs Business Hours')
    
    # 8. Feature importance bar chart
    ax9 = plt.subplot(3, 3, 9)
    exclude_cols = ['transaction_id', 'user_id', 'timestamp', 'is_fraud']
    feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in exclude_cols]
    correlations = df[feature_cols + ['is_fraud']].corr()['is_fraud'].drop('is_fraud')
    top_features = correlations.abs().sort_values(ascending=True).tail(10)
    ax9.barh(range(len(top_features)), top_features.values)
    ax9.set_yticks(range(len(top_features)))
    ax9.set_yticklabels(top_features.index, fontsize=8)
    ax9.set_xlabel('Abs. Correlation with Fraud')
    ax9.set_title('Top 10 Features by Correlation')
    ax9.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'fraud_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualizations saved to: {output_file}")
    
    # Show the plot
    plt.show()


def prepare_for_ml(df):
    """Prepare data for machine learning"""
    print("\n" + "=" * 70)
    print("PREPARING DATA FOR MACHINE LEARNING")
    print("=" * 70)
    
    # Separate features and target
    exclude_cols = ['transaction_id', 'user_id', 'user_profile', 
                   'timestamp', 'fraud_pattern', 'is_fraud']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['is_fraud']
    
    print(f"\n📊 ML-Ready Dataset:")
    print(f"   Features (X): {X.shape}")
    print(f"   Target (y):   {y.shape}")
    print(f"\n   Class distribution:")
    print(f"   Normal: {(y == 0).sum()}")
    print(f"   Fraud:  {(y == 1).sum()}")
    
    # Show feature names
    print(f"\n🔧 Features ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, 1):
        print(f"   {i:2d}. {col}")
        if i >= 20:
            print(f"   ... and {len(feature_cols) - 20} more")
            break
    
    print(f"\n✅ Data ready for model training!")
    print(f"   Use these features: X")
    print(f"   Use this target: y")
    
    return X, y


def main():
    """Main execution"""
    print("\n")
    print("=" * 70)
    print("FRAUD DETECTION DATASET ANALYSIS")
    print("=" * 70)
    
    # Find latest dataset
    filename = find_latest_dataset()
    if filename is None:
        return 1
    
    # Load and explore
    df = load_and_explore(filename)
    
    # Analyze fraud patterns
    analyze_fraud_patterns(df)
    
    # Feature correlation analysis
    correlations = analyze_feature_importance(df)
    
    # Create visualizations
    visualize_data(df)
    
    # Prepare for ML
    X, y = prepare_for_ml(df)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\n🎯 Next Steps:")
    print("   1. Review the fraud_analysis.png visualization")
    print("   2. Use this dataset to train your PyTorch model")
    print("   3. Focus on high-correlation features for better accuracy")
    print("   4. Handle class imbalance (consider SMOTE or class weights)")
    print("=" * 70)
    print()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
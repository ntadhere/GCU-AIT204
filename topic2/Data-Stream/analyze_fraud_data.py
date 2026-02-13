#!/usr/bin/env python3
"""
Fraud Detection Dataset Analysis
=================================
Comprehensive analysis of collected fraud detection data including:
- Fraud rate calculation
- Pattern identification
- Visualizations of fraud vs normal transactions
- Time pattern analysis
- User profile analysis

Usage:
    python3 analyze_fraud_data.py fraud_data_20260130_143522.csv
    python3 analyze_fraud_data.py  # Auto-finds latest dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from datetime import datetime
import json


class FraudAnalyzer:
    """Comprehensive fraud detection data analyzer"""
    
    def __init__(self, filename=None):
        """Initialize analyzer with dataset"""
        if filename is None:
            filename = self.find_latest_dataset()
        
        if filename is None:
            raise FileNotFoundError("No fraud dataset found. Run fraud_collector.py first.")
        
        self.filename = filename
        self.df = pd.read_csv(filename)
        self.fraud_df = self.df[self.df['is_fraud'] == 1]
        self.normal_df = self.df[self.df['is_fraud'] == 0]
        
        print("=" * 80)
        print("FRAUD DETECTION DATASET ANALYSIS")
        print("=" * 80)
        print(f"\n📁 Dataset: {filename}")
        print(f"📊 Total transactions: {len(self.df):,}")
        print(f"📅 Loaded successfully!")
    
    def find_latest_dataset(self):
        """Find the most recently created fraud dataset"""
        data_files = list(Path('.').glob('fraud_data_*.csv'))
        if not data_files:
            return None
        return max(data_files, key=lambda p: p.stat().st_mtime)
    
    def calculate_fraud_rate(self):
        """Task 1: Calculate and display fraud rate"""
        print("\n" + "=" * 80)
        print("TASK 1: FRAUD RATE CALCULATION")
        print("=" * 80)
        
        total = len(self.df)
        fraud_count = len(self.fraud_df)
        normal_count = len(self.normal_df)
        fraud_rate = (fraud_count / total) * 100
        
        print(f"\n📊 Overall Statistics:")
        print(f"   Total Transactions:      {total:,}")
        print(f"   Fraudulent Transactions: {fraud_count:,}")
        print(f"   Normal Transactions:     {normal_count:,}")
        print(f"\n🚨 FRAUD RATE: {fraud_rate:.2f}%")
        
        # Class balance analysis
        print(f"\n⚖️  Class Balance:")
        print(f"   Normal: {(normal_count/total)*100:.2f}%")
        print(f"   Fraud:  {(fraud_count/total)*100:.2f}%")
        
        if fraud_rate < 5:
            print(f"\n   ⚠️  Low fraud rate - may need to collect more data")
        elif fraud_rate > 15:
            print(f"\n   ⚠️  High fraud rate - unusual, check data collection")
        else:
            print(f"\n   ✅ Good fraud rate for training (~5-15% is typical)")
        
        # Fraud rate by dataset size
        fraud_rates_by_size = []
        sizes = np.linspace(100, len(self.df), 10).astype(int)
        for size in sizes:
            subset_fraud_rate = (self.df.iloc[:size]['is_fraud'].sum() / size) * 100
            fraud_rates_by_size.append(subset_fraud_rate)
        
        print(f"\n📈 Fraud Rate Stability:")
        print(f"   First 100 transactions: {fraud_rates_by_size[0]:.2f}%")
        print(f"   First 500 transactions: {fraud_rates_by_size[4]:.2f}%")
        print(f"   All transactions:       {fraud_rate:.2f}%")
        
        return {
            'total': total,
            'fraud_count': fraud_count,
            'normal_count': normal_count,
            'fraud_rate': fraud_rate
        }
    
    def identify_patterns(self):
        """Task 2: Identify patterns in fraudulent vs normal transactions"""
        print("\n" + "=" * 80)
        print("TASK 2: PATTERN IDENTIFICATION")
        print("=" * 80)
        
        patterns = {}
        
        # 1. Amount patterns
        print("\n💰 AMOUNT PATTERNS:")
        print(f"\n   Normal Transactions:")
        print(f"      Mean:   ${self.normal_df['amount'].mean():,.2f}")
        print(f"      Median: ${self.normal_df['amount'].median():,.2f}")
        print(f"      Std:    ${self.normal_df['amount'].std():,.2f}")
        print(f"      Min:    ${self.normal_df['amount'].min():,.2f}")
        print(f"      Max:    ${self.normal_df['amount'].max():,.2f}")
        
        print(f"\n   Fraudulent Transactions:")
        print(f"      Mean:   ${self.fraud_df['amount'].mean():,.2f}")
        print(f"      Median: ${self.fraud_df['amount'].median():,.2f}")
        print(f"      Std:    ${self.fraud_df['amount'].std():,.2f}")
        print(f"      Min:    ${self.fraud_df['amount'].min():,.2f}")
        print(f"      Max:    ${self.fraud_df['amount'].max():,.2f}")
        
        amount_diff_pct = ((self.fraud_df['amount'].mean() - self.normal_df['amount'].mean()) / 
                          self.normal_df['amount'].mean()) * 100
        print(f"\n   🔍 Fraud transactions are {abs(amount_diff_pct):.1f}% {'higher' if amount_diff_pct > 0 else 'lower'} on average")
        
        patterns['amount'] = {
            'normal_mean': self.normal_df['amount'].mean(),
            'fraud_mean': self.fraud_df['amount'].mean(),
            'difference_pct': amount_diff_pct
        }
        
        # 2. Time patterns
        print("\n⏰ TIME PATTERNS:")
        
        # Hour distribution
        normal_peak_hour = self.normal_df['hour_of_day'].mode().values[0]
        fraud_peak_hour = self.fraud_df['hour_of_day'].mode().values[0]
        
        print(f"\n   Peak Hours:")
        print(f"      Normal transactions: {int(normal_peak_hour)}:00")
        print(f"      Fraud transactions:  {int(fraud_peak_hour)}:00")
        
        # Night transactions
        normal_night_pct = (self.normal_df['is_night'].sum() / len(self.normal_df)) * 100
        fraud_night_pct = (self.fraud_df['is_night'].sum() / len(self.fraud_df)) * 100
        
        print(f"\n   Night Transactions (10pm-5am):")
        print(f"      Normal: {normal_night_pct:.1f}%")
        print(f"      Fraud:  {fraud_night_pct:.1f}%")
        if fraud_night_pct > normal_night_pct * 1.5:
            print(f"      🚨 Fraud is {fraud_night_pct/normal_night_pct:.1f}x more likely at night!")
        
        # Weekend patterns
        normal_weekend_pct = (self.normal_df['is_weekend'].sum() / len(self.normal_df)) * 100
        fraud_weekend_pct = (self.fraud_df['is_weekend'].sum() / len(self.fraud_df)) * 100
        
        print(f"\n   Weekend Transactions:")
        print(f"      Normal: {normal_weekend_pct:.1f}%")
        print(f"      Fraud:  {fraud_weekend_pct:.1f}%")
        
        patterns['time'] = {
            'fraud_night_multiplier': fraud_night_pct / normal_night_pct if normal_night_pct > 0 else 0,
            'fraud_weekend_pct': fraud_weekend_pct
        }
        
        # 3. User behavior patterns
        print("\n👤 USER BEHAVIOR PATTERNS:")
        
        # Amount deviation
        normal_dev = self.normal_df['amount_deviation'].abs().mean()
        fraud_dev = self.fraud_df['amount_deviation'].abs().mean()
        
        print(f"\n   Average Deviation from User's Normal:")
        print(f"      Normal transactions: {normal_dev:.2f} ({normal_dev*100:.0f}%)")
        print(f"      Fraud transactions:  {fraud_dev:.2f} ({fraud_dev*100:.0f}%)")
        if fraud_dev > normal_dev * 2:
            print(f"      🚨 Fraud deviates {fraud_dev/normal_dev:.1f}x more from user's normal behavior!")
        
        # Z-score analysis
        normal_zscore = self.normal_df['amount_zscore'].abs().mean()
        fraud_zscore = self.fraud_df['amount_zscore'].abs().mean()
        
        print(f"\n   Average Amount Z-Score:")
        print(f"      Normal transactions: {normal_zscore:.2f}")
        print(f"      Fraud transactions:  {fraud_zscore:.2f}")
        print(f"      (Z-score > 3 is highly unusual)")
        
        # High z-score analysis
        high_zscore_normal = (self.normal_df['amount_zscore'].abs() > 3).sum()
        high_zscore_fraud = (self.fraud_df['amount_zscore'].abs() > 3).sum()
        
        print(f"\n   Transactions with |Z-score| > 3:")
        print(f"      Normal: {high_zscore_normal} ({(high_zscore_normal/len(self.normal_df))*100:.1f}%)")
        print(f"      Fraud:  {high_zscore_fraud} ({(high_zscore_fraud/len(self.fraud_df))*100:.1f}%)")
        
        patterns['behavior'] = {
            'fraud_zscore_mean': fraud_zscore,
            'normal_zscore_mean': normal_zscore
        }
        
        # 4. Rapid transactions
        print("\n⚡ VELOCITY PATTERNS:")
        normal_rapid = (self.normal_df['rapid_transaction'].sum() / len(self.normal_df)) * 100
        fraud_rapid = (self.fraud_df['rapid_transaction'].sum() / len(self.fraud_df)) * 100
        
        print(f"\n   Rapid Transactions (same day):")
        print(f"      Normal: {normal_rapid:.1f}%")
        print(f"      Fraud:  {fraud_rapid:.1f}%")
        if fraud_rapid > normal_rapid * 2:
            print(f"      🚨 Fraud is {fraud_rapid/normal_rapid:.1f}x more likely in rapid succession!")
        
        # 5. Round numbers
        print("\n💯 ROUND NUMBER PATTERNS:")
        normal_round = (self.normal_df['is_round_number'].sum() / len(self.normal_df)) * 100
        fraud_round = (self.fraud_df['is_round_number'].sum() / len(self.fraud_df)) * 100
        
        print(f"\n   Round Number Amounts ($50, $100, etc.):")
        print(f"      Normal: {normal_round:.1f}%")
        print(f"      Fraud:  {fraud_round:.1f}%")
        if fraud_round > normal_round * 2:
            print(f"      🚨 Fraud is {fraud_round/normal_round:.1f}x more likely to be round numbers!")
        
        # 6. Fraud pattern distribution
        if 'fraud_pattern' in self.df.columns:
            print("\n🎯 FRAUD ATTACK PATTERNS:")
            pattern_counts = self.fraud_df['fraud_pattern'].value_counts()
            print(f"\n   Pattern                  Count    Percentage")
            print(f"   {'─'*50}")
            for pattern, count in pattern_counts.items():
                pct = (count / len(self.fraud_df)) * 100
                bar = '█' * int(pct / 5)
                print(f"   {pattern:20s} {count:5d}    {pct:5.1f}% {bar}")
        
        return patterns
    
    def create_visualizations(self):
        """Task 3: Create comprehensive visualizations"""
        print("\n" + "=" * 80)
        print("TASK 3: CREATING VISUALIZATIONS")
        print("=" * 80)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle('Fraud Detection Analysis: Patterns and Insights', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)
        
        # ========== ROW 1: Amount Distributions ==========
        
        # 1. Normal transaction amounts
        ax1 = fig.add_subplot(gs[0, 0:2])
        self.normal_df['amount'].hist(bins=40, alpha=0.7, color='green', 
                                       edgecolor='black', ax=ax1)
        ax1.axvline(self.normal_df['amount'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f'Mean: ${self.normal_df["amount"].mean():.2f}')
        ax1.set_xlabel('Transaction Amount ($)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Normal Transaction Amount Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Fraudulent transaction amounts
        ax2 = fig.add_subplot(gs[0, 2:4])
        self.fraud_df['amount'].hist(bins=40, alpha=0.7, color='red', 
                                      edgecolor='black', ax=ax2)
        ax2.axvline(self.fraud_df['amount'].mean(), color='darkred', 
                   linestyle='--', linewidth=2, label=f'Mean: ${self.fraud_df["amount"].mean():.2f}')
        ax2.set_xlabel('Transaction Amount ($)', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Fraudulent Transaction Amount Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # ========== ROW 2: Time Patterns ==========
        
        # 3. Fraud rate by hour of day
        ax3 = fig.add_subplot(gs[1, 0:2])
        fraud_by_hour = self.df.groupby('hour_of_day').agg({
            'is_fraud': ['sum', 'count', 'mean']
        })
        fraud_by_hour.columns = ['fraud_count', 'total_count', 'fraud_rate']
        fraud_by_hour['fraud_rate'] *= 100
        
        bars = ax3.bar(fraud_by_hour.index, fraud_by_hour['fraud_rate'], 
                      color=['red' if x > fraud_by_hour['fraud_rate'].mean() else 'orange' 
                            for x in fraud_by_hour['fraud_rate']],
                      edgecolor='black', alpha=0.7)
        ax3.axhline(fraud_by_hour['fraud_rate'].mean(), color='blue', 
                   linestyle='--', linewidth=2, label=f'Average: {fraud_by_hour["fraud_rate"].mean():.1f}%')
        ax3.set_xlabel('Hour of Day', fontsize=11)
        ax3.set_ylabel('Fraud Rate (%)', fontsize=11)
        ax3.set_title('Fraud Rate by Hour of Day', fontsize=12, fontweight='bold')
        ax3.set_xticks(range(0, 24, 2))
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Transaction count by hour (stacked)
        ax4 = fig.add_subplot(gs[1, 2:4])
        hour_fraud = self.fraud_df.groupby('hour_of_day').size()
        hour_normal = self.normal_df.groupby('hour_of_day').size()
        hours = range(24)
        
        # Ensure all hours are represented
        hour_fraud = hour_fraud.reindex(hours, fill_value=0)
        hour_normal = hour_normal.reindex(hours, fill_value=0)
        
        ax4.bar(hours, hour_normal, label='Normal', color='green', alpha=0.7, edgecolor='black')
        ax4.bar(hours, hour_fraud, bottom=hour_normal, label='Fraud', 
               color='red', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Hour of Day', fontsize=11)
        ax4.set_ylabel('Number of Transactions', fontsize=11)
        ax4.set_title('Transaction Volume by Hour (Stacked)', fontsize=12, fontweight='bold')
        ax4.set_xticks(range(0, 24, 2))
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # ========== ROW 3: User Profiles and Patterns ==========
        
        # 5. Top user profiles by fraud rate
        ax5 = fig.add_subplot(gs[2, 0:2])
        if 'user_profile' in self.df.columns:
            profile_stats = self.df.groupby('user_profile').agg({
                'is_fraud': ['sum', 'count', 'mean']
            })
            profile_stats.columns = ['fraud_count', 'total_count', 'fraud_rate']
            profile_stats['fraud_rate'] *= 100
            profile_stats = profile_stats[profile_stats['total_count'] >= 10]  # Min 10 transactions
            profile_stats = profile_stats.sort_values('fraud_rate', ascending=False).head(15)
            
            y_pos = np.arange(len(profile_stats))
            colors = ['darkred' if 'fraud_prone' in idx else 'orange' 
                     for idx in profile_stats.index]
            
            ax5.barh(y_pos, profile_stats['fraud_rate'], color=colors, 
                    edgecolor='black', alpha=0.7)
            ax5.set_yticks(y_pos)
            ax5.set_yticklabels(profile_stats.index, fontsize=9)
            ax5.set_xlabel('Fraud Rate (%)', fontsize=11)
            ax5.set_title('Top 15 User Profiles by Fraud Rate', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='x')
            
            # Add count labels
            for i, (idx, row) in enumerate(profile_stats.iterrows()):
                ax5.text(row['fraud_rate'] + 0.5, i, 
                        f"{int(row['fraud_count'])}/{int(row['total_count'])}", 
                        va='center', fontsize=8)
        
        # 6. Fraud patterns pie chart
        ax6 = fig.add_subplot(gs[2, 2:4])
        if 'fraud_pattern' in self.df.columns and len(self.fraud_df) > 0:
            pattern_counts = self.fraud_df['fraud_pattern'].value_counts()
            colors_pie = plt.cm.Reds(np.linspace(0.4, 0.8, len(pattern_counts)))
            
            wedges, texts, autotexts = ax6.pie(pattern_counts.values, 
                                               labels=pattern_counts.index,
                                               autopct='%1.1f%%',
                                               colors=colors_pie,
                                               startangle=90,
                                               textprops={'fontsize': 9})
            ax6.set_title('Distribution of Fraud Attack Patterns', fontsize=12, fontweight='bold')
            
            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        # ========== ROW 4: Advanced Analysis ==========
        
        # 7. Amount Z-Score comparison
        ax7 = fig.add_subplot(gs[3, 0])
        zscore_data = [self.normal_df['amount_zscore'].abs(), 
                      self.fraud_df['amount_zscore'].abs()]
        bp = ax7.boxplot(zscore_data, labels=['Normal', 'Fraud'],
                        patch_artist=True, showfliers=False)
        bp['boxes'][0].set_facecolor('green')
        bp['boxes'][1].set_facecolor('red')
        for box in bp['boxes']:
            box.set_alpha(0.7)
        ax7.axhline(y=3, color='blue', linestyle='--', linewidth=2, 
                   label='Z=3 (Highly Unusual)', alpha=0.7)
        ax7.set_ylabel('Absolute Z-Score', fontsize=11)
        ax7.set_title('Amount Z-Score Distribution', fontsize=12, fontweight='bold')
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3, axis='y')
        
        # 8. Weekend vs Weekday fraud
        ax8 = fig.add_subplot(gs[3, 1])
        weekend_stats = self.df.groupby('is_weekend')['is_fraud'].agg(['sum', 'count', 'mean'])
        weekend_stats['fraud_rate'] = weekend_stats['mean'] * 100
        
        bars = ax8.bar(['Weekday', 'Weekend'], weekend_stats['fraud_rate'], 
                      color=['skyblue', 'coral'], edgecolor='black', alpha=0.7)
        ax8.set_ylabel('Fraud Rate (%)', fontsize=11)
        ax8.set_title('Fraud Rate:\nWeekday vs Weekend', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 9. Feature correlation heatmap (top features)
        ax9 = fig.add_subplot(gs[3, 2:4])
        
        # Select top correlated features
        exclude_cols = ['transaction_id', 'user_id', 'timestamp', 'is_fraud']
        feature_cols = [col for col in self.df.select_dtypes(include=[np.number]).columns 
                       if col not in exclude_cols]
        
        correlations = self.df[feature_cols + ['is_fraud']].corr()['is_fraud'].drop('is_fraud')
        top_features = correlations.abs().sort_values(ascending=False).head(10).index.tolist()
        
        # Create correlation matrix for top features
        corr_matrix = self.df[top_features + ['is_fraud']].corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                   center=0, square=True, ax=ax9, cbar_kws={'shrink': 0.8},
                   annot_kws={'fontsize': 8})
        ax9.set_title('Top 10 Features Correlation with Fraud', fontsize=12, fontweight='bold')
        plt.setp(ax9.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        plt.setp(ax9.get_yticklabels(), rotation=0, fontsize=9)
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'fraud_analysis_{timestamp}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✅ Comprehensive visualization saved: {output_file}")
        
        # Create second figure for additional insights
        self._create_additional_visualizations(timestamp)
        
        plt.show()
        
        return output_file
    
    def _create_additional_visualizations(self, timestamp):
        """Create additional detailed visualizations"""
        fig2 = plt.figure(figsize=(16, 10))
        fig2.suptitle('Detailed Fraud Analysis: User Behavior & Patterns', 
                     fontsize=16, fontweight='bold')
        
        gs2 = fig2.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Amount deviation scatter
        ax1 = fig2.add_subplot(gs2[0, :2])
        sample_size = min(1000, len(self.df))
        sample_df = self.df.sample(sample_size) if len(self.df) > sample_size else self.df
        
        normal_sample = sample_df[sample_df['is_fraud'] == 0]
        fraud_sample = sample_df[sample_df['is_fraud'] == 1]
        
        ax1.scatter(normal_sample['amount'], normal_sample['amount_deviation'], 
                   alpha=0.4, s=20, c='green', label='Normal', edgecolors='black', linewidth=0.5)
        ax1.scatter(fraud_sample['amount'], fraud_sample['amount_deviation'], 
                   alpha=0.7, s=50, c='red', label='Fraud', edgecolors='black', linewidth=0.5)
        ax1.axhline(y=0, color='blue', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Transaction Amount ($)')
        ax1.set_ylabel('Deviation from User Average')
        ax1.set_title('Amount vs User Behavior Deviation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. User transaction count distribution
        ax2 = fig2.add_subplot(gs2[0, 2])
        user_fraud_counts = self.df.groupby('user_id')['is_fraud'].agg(['sum', 'count'])
        user_fraud_counts['fraud_rate'] = (user_fraud_counts['sum'] / user_fraud_counts['count']) * 100
        
        ax2.hist(user_fraud_counts['fraud_rate'], bins=20, color='purple', 
                alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Fraud Rate (%)')
        ax2.set_ylabel('Number of Users')
        ax2.set_title('Distribution of User Fraud Rates')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Time heatmap
        ax3 = fig2.add_subplot(gs2[1, :])
        
        # Create hour x day_of_week heatmap
        self.df['day_of_week'] = pd.to_datetime(self.df['timestamp']).dt.dayofweek
        fraud_heatmap = self.df.groupby(['day_of_week', 'hour_of_day'])['is_fraud'].mean() * 100
        fraud_heatmap = fraud_heatmap.unstack(fill_value=0)
        
        sns.heatmap(fraud_heatmap, cmap='YlOrRd', annot=False, fmt='.1f', 
                   cbar_kws={'label': 'Fraud Rate (%)'}, ax=ax3)
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Day of Week')
        ax3.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
        ax3.set_title('Fraud Rate Heatmap: Day of Week vs Hour of Day')
        
        # 4. Top fraudulent users
        ax4 = fig2.add_subplot(gs2[2, 0])
        top_fraud_users = self.df.groupby('user_id')['is_fraud'].sum().sort_values(ascending=False).head(10)
        
        ax4.barh(range(len(top_fraud_users)), top_fraud_users.values, 
                color='darkred', alpha=0.7, edgecolor='black')
        ax4.set_yticks(range(len(top_fraud_users)))
        ax4.set_yticklabels([f'User {uid}' for uid in top_fraud_users.index])
        ax4.set_xlabel('Number of Fraudulent Transactions')
        ax4.set_title('Top 10 Users by Fraud Count')
        ax4.grid(True, alpha=0.3, axis='x')
        
        # 5. Merchant analysis
        ax5 = fig2.add_subplot(gs2[2, 1])
        if 'merchant_id' in self.df.columns:
            merchant_fraud = self.df.groupby('merchant_id')['is_fraud'].agg(['sum', 'count', 'mean'])
            merchant_fraud = merchant_fraud[merchant_fraud['count'] >= 5]  # Min 5 transactions
            merchant_fraud = merchant_fraud.sort_values('mean', ascending=False).head(10)
            
            ax5.barh(range(len(merchant_fraud)), merchant_fraud['mean'] * 100,
                    color='orange', alpha=0.7, edgecolor='black')
            ax5.set_yticks(range(len(merchant_fraud)))
            ax5.set_yticklabels([f'M{mid}' for mid in merchant_fraud.index])
            ax5.set_xlabel('Fraud Rate (%)')
            ax5.set_title('Top 10 Merchants by Fraud Rate')
            ax5.grid(True, alpha=0.3, axis='x')
        
        # 6. Feature importance
        ax6 = fig2.add_subplot(gs2[2, 2])
        exclude_cols = ['transaction_id', 'user_id', 'timestamp', 'is_fraud']
        feature_cols = [col for col in self.df.select_dtypes(include=[np.number]).columns 
                       if col not in exclude_cols]
        
        correlations = self.df[feature_cols + ['is_fraud']].corr()['is_fraud'].drop('is_fraud')
        top_corr = correlations.abs().sort_values(ascending=True).tail(10)
        
        colors = ['red' if x > 0 else 'blue' for x in correlations[top_corr.index]]
        ax6.barh(range(len(top_corr)), top_corr.values, color=colors, alpha=0.7, edgecolor='black')
        ax6.set_yticks(range(len(top_corr)))
        ax6.set_yticklabels(top_corr.index, fontsize=9)
        ax6.set_xlabel('Absolute Correlation')
        ax6.set_title('Top 10 Predictive Features')
        ax6.grid(True, alpha=0.3, axis='x')
        
        output_file2 = f'fraud_analysis_detailed_{timestamp}.png'
        plt.savefig(output_file2, dpi=150, bbox_inches='tight')
        print(f"✅ Detailed visualization saved: {output_file2}")
    
    def generate_report(self):
        """Generate a comprehensive text report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f'fraud_analysis_report_{timestamp}.txt'
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FRAUD DETECTION DATASET ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.filename}\n")
            f.write(f"Total Transactions: {len(self.df):,}\n")
            
            # Summary statistics
            f.write("\n" + "=" * 80 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("=" * 80 + "\n")
            f.write(f"\nFraud Rate: {(len(self.fraud_df)/len(self.df))*100:.2f}%\n")
            f.write(f"Fraudulent Transactions: {len(self.fraud_df):,}\n")
            f.write(f"Normal Transactions: {len(self.normal_df):,}\n")
            
            # Key findings
            f.write("\n" + "=" * 80 + "\n")
            f.write("KEY FINDINGS\n")
            f.write("=" * 80 + "\n")
            
            # Amount patterns
            f.write(f"\n1. Amount Patterns:\n")
            f.write(f"   - Average fraud transaction: ${self.fraud_df['amount'].mean():.2f}\n")
            f.write(f"   - Average normal transaction: ${self.normal_df['amount'].mean():.2f}\n")
            f.write(f"   - Fraud transactions are {((self.fraud_df['amount'].mean() / self.normal_df['amount'].mean()) - 1) * 100:.1f}% larger\n")
            
            # Time patterns
            fraud_night_pct = (self.fraud_df['is_night'].sum() / len(self.fraud_df)) * 100
            normal_night_pct = (self.normal_df['is_night'].sum() / len(self.normal_df)) * 100
            
            f.write(f"\n2. Time Patterns:\n")
            f.write(f"   - Fraud at night: {fraud_night_pct:.1f}%\n")
            f.write(f"   - Normal at night: {normal_night_pct:.1f}%\n")
            f.write(f"   - Night fraud multiplier: {fraud_night_pct/normal_night_pct if normal_night_pct > 0 else 0:.2f}x\n")
            
            # User profiles
            if 'user_profile' in self.df.columns:
                f.write(f"\n3. Top Fraud-Prone Profiles:\n")
                profile_fraud = self.df.groupby('user_profile')['is_fraud'].mean().sort_values(ascending=False).head(5)
                for profile, rate in profile_fraud.items():
                    f.write(f"   - {profile}: {rate*100:.1f}% fraud rate\n")
            
            # Fraud patterns
            if 'fraud_pattern' in self.df.columns:
                f.write(f"\n4. Most Common Fraud Patterns:\n")
                patterns = self.fraud_df['fraud_pattern'].value_counts().head(5)
                for pattern, count in patterns.items():
                    f.write(f"   - {pattern}: {count} instances ({(count/len(self.fraud_df))*100:.1f}%)\n")
        
        print(f"✅ Text report saved: {report_file}")
        return report_file
    
    def run_complete_analysis(self):
        """Run all analysis tasks"""
        print("\n🚀 Starting complete fraud analysis...\n")
        
        # Task 1: Calculate fraud rate
        fraud_stats = self.calculate_fraud_rate()
        
        # Task 2: Identify patterns
        patterns = self.identify_patterns()
        
        # Task 3: Create visualizations
        viz_file = self.create_visualizations()
        
        # Generate report
        report_file = self.generate_report()
        
        # Summary
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\n📊 Results Summary:")
        print(f"   • Total transactions analyzed: {len(self.df):,}")
        print(f"   • Fraud rate: {fraud_stats['fraud_rate']:.2f}%")
        print(f"   • Visualizations created: 2 files")
        print(f"   • Report generated: 1 file")
        
        print(f"\n📁 Output Files:")
        print(f"   • Main visualization: fraud_analysis_*.png")
        print(f"   • Detailed visualization: fraud_analysis_detailed_*.png")
        print(f"   • Text report: {report_file}")
        
        print(f"\n🎯 Key Insights:")
        print(f"   • Fraud transactions average ${self.fraud_df['amount'].mean():.2f}")
        print(f"   • Normal transactions average ${self.normal_df['amount'].mean():.2f}")
        
        if 'amount_zscore' in self.df.columns:
            fraud_high_z = (self.fraud_df['amount_zscore'].abs() > 3).sum()
            print(f"   • {fraud_high_z} fraud transactions with z-score > 3 (highly unusual)")
        
        print("\n" + "=" * 80)
        
        return {
            'fraud_stats': fraud_stats,
            'patterns': patterns,
            'visualization': viz_file,
            'report': report_file
        }


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Comprehensive Fraud Detection Dataset Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-find latest dataset
    python3 analyze_fraud_data.py
    
    # Analyze specific dataset
    python3 analyze_fraud_data.py fraud_data_20260130_143522.csv
    
    # Just calculate fraud rate
    python3 analyze_fraud_data.py --quick
        """
    )
    
    parser.add_argument('dataset', nargs='?', default=None,
                       help='Path to fraud dataset CSV file (auto-detects if not provided)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick analysis (fraud rate only, no visualizations)')
    
    args = parser.parse_args()
    
    try:
        # Create analyzer
        analyzer = FraudAnalyzer(args.dataset)
        
        if args.quick:
            # Quick mode: just fraud rate
            analyzer.calculate_fraud_rate()
        else:
            # Full analysis
            analyzer.run_complete_analysis()
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Please collect data first:")
        print("   python3 fraud_collector.py --size 2000")
        return 1
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
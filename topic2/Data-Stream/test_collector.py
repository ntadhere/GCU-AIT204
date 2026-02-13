#!/usr/bin/env python3
"""
Quick test script for the fraud collector.
This simulates what would happen when collecting data.
"""

import json
import pandas as pd


def test_feature_extraction():
    """Test that feature extraction works correctly"""
    print("=" * 70)
    print("TESTING FRAUD COLLECTOR - FEATURE EXTRACTION")
    print("=" * 70)
    
    # Simulate a transaction from the fraud server
    sample_transaction = {
        "transactionID": 1,
        "userID": 13,
        "amount": 459.99,
        "timestamp": "2024-01-15T14:30:45",
        "itemID": "E5678",
        "merchantID": "M234",
        "isWeekend": False,
        "hourOfDay": 14,
        "daysSinceLastTransaction": 1,
        "userProfile": "fraud_prone_1",
        "isFraud": True,
        "fraudPattern": "sudden_spike"
    }
    
    print("\n📥 Sample Transaction:")
    print(json.dumps(sample_transaction, indent=2))
    
    # Import the collector
    from fraud_collector import FraudDataCollector
    
    # Create collector instance
    collector = FraudDataCollector(collection_size=10)
    
    # Simulate collecting this transaction
    collector.transactions.append(sample_transaction)
    collector._update_user_stats(sample_transaction)
    
    # Extract features
    features = collector.extract_features(sample_transaction)
    
    print("\n🔧 Extracted Features:")
    print(f"Total features: {len(features)}")
    
    # Display features in categories
    print("\n📊 Basic Features:")
    basic = ['amount', 'amount_log', 'hour_of_day', 'is_weekend', 'days_since_last_transaction']
    for feat in basic:
        if feat in features:
            print(f"  {feat:30s}: {features[feat]}")
    
    print("\n⏰ Time Pattern Features:")
    time_feats = ['is_night', 'is_business_hours', 'is_lunch_time', 'is_evening', 'hour_sin', 'hour_cos']
    for feat in time_feats:
        if feat in features:
            print(f"  {feat:30s}: {features[feat]:.4f}")
    
    print("\n🏷️ Category Features:")
    category_feats = [k for k in features.keys() if k.startswith('item_category_')]
    for feat in sorted(category_feats):
        print(f"  {feat:30s}: {features[feat]}")
    
    print("\n👤 User Behavior Features:")
    user_feats = ['user_transaction_count', 'user_avg_amount', 'user_std_amount', 
                  'amount_deviation', 'amount_zscore']
    for feat in user_feats:
        if feat in features:
            print(f"  {feat:30s}: {features[feat]:.4f}")
    
    print("\n🚨 Fraud Indicators:")
    fraud_feats = ['is_round_number', 'rapid_transaction', 'is_fraud']
    for feat in fraud_feats:
        if feat in features:
            print(f"  {feat:30s}: {features[feat]}")
    
    print("\n📋 Metadata:")
    meta_feats = ['transaction_id', 'user_id', 'user_profile', 'fraud_pattern']
    for feat in meta_feats:
        if feat in features:
            print(f"  {feat:30s}: {features[feat]}")
    
    print("\n" + "=" * 70)
    print("✓ Feature extraction test passed!")
    print("=" * 70)
    
    return features


def test_dataframe_creation():
    """Test that we can create a proper DataFrame"""
    print("\n" + "=" * 70)
    print("TESTING DATAFRAME CREATION")
    print("=" * 70)
    
    from fraud_collector import FraudDataCollector
    
    # Create sample transactions
    sample_transactions = [
        {
            "transactionID": 1, "userID": 5, "amount": 23.50,
            "timestamp": "2024-01-15T12:30:00", "itemID": "C1234",
            "merchantID": "M100", "isWeekend": False, "hourOfDay": 12,
            "daysSinceLastTransaction": 1, "userProfile": "student",
            "isFraud": False
        },
        {
            "transactionID": 2, "userID": 13, "amount": 450.00,
            "timestamp": "2024-01-15T03:15:00", "itemID": "E9999",
            "merchantID": "M999", "isWeekend": False, "hourOfDay": 3,
            "daysSinceLastTransaction": 0, "userProfile": "fraud_prone_1",
            "isFraud": True, "fraudPattern": "sudden_spike"
        },
        {
            "transactionID": 3, "userID": 5, "amount": 27.80,
            "timestamp": "2024-01-15T18:45:00", "itemID": "B5678",
            "merchantID": "M100", "isWeekend": False, "hourOfDay": 18,
            "daysSinceLastTransaction": 0, "userProfile": "student",
            "isFraud": False
        }
    ]
    
    collector = FraudDataCollector(collection_size=len(sample_transactions))
    
    # Process transactions
    for trans in sample_transactions:
        collector.transactions.append(trans)
        collector._update_user_stats(trans)
    
    # Extract features
    feature_list = []
    for trans in sample_transactions:
        features = collector.extract_features(trans)
        feature_list.append(features)
    
    # Create DataFrame
    df = pd.DataFrame(feature_list)
    
    print(f"\n📊 DataFrame created successfully!")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {len(df.columns)}")
    
    print("\n🔍 First few rows:")
    display_cols = ['transaction_id', 'user_id', 'amount', 'hour_of_day', 
                    'is_night', 'amount_deviation', 'is_fraud']
    print(df[display_cols].to_string(index=False))
    
    print("\n📈 Class Distribution:")
    print(df['is_fraud'].value_counts())
    
    print("\n" + "=" * 70)
    print("✓ DataFrame creation test passed!")
    print("=" * 70)
    
    return df


def show_usage_examples():
    """Show examples of how to use the collector"""
    print("\n" + "=" * 70)
    print("USAGE EXAMPLES")
    print("=" * 70)
    
    examples = [
        ("Quick test (100 transactions)", 
         "python3 fraud_collector.py --size 100"),
        
        ("Standard training set (2000 transactions)", 
         "python3 fraud_collector.py --size 2000"),
        
        ("Large dataset (5000 transactions)", 
         "python3 fraud_collector.py --size 5000 --output large_dataset.csv"),
        
        ("Connect to remote server", 
         "python3 fraud_collector.py --host 192.168.1.10 --port 5555 --size 2000"),
    ]
    
    for i, (desc, cmd) in enumerate(examples, 1):
        print(f"\n{i}. {desc}")
        print(f"   {cmd}")
    
    print("\n" + "=" * 70)


def main():
    """Run all tests"""
    print("\n")
    print("🧪 " * 20)
    print("FRAUD COLLECTOR TEST SUITE")
    print("🧪 " * 20)
    
    try:
        # Test 1: Feature extraction
        features = test_feature_extraction()
        
        # Test 2: DataFrame creation
        df = test_dataframe_creation()
        
        # Show usage examples
        show_usage_examples()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\n🎯 Next steps:")
        print("   1. Start the fraud stream server:")
        print("      python3 fraud_stream_server.py")
        print("")
        print("   2. Run the collector:")
        print("      python3 fraud_collector.py --size 2000")
        print("")
        print("   3. Start building your fraud detection model!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
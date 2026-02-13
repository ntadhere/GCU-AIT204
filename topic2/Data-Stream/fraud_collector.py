#!/usr/bin/env python3
"""
Fraud Data Collector
====================
Connects to the fraud stream server and collects labeled transaction data
for training fraud detection models.

Usage:
    python fraud_collector.py --size 1000 --host localhost --port 5555
    python fraud_collector.py --size 5000 --output custom_dataset.csv
"""

import json
import socket
import pandas as pd
from datetime import datetime
import pickle
import sys
import argparse
from collections import defaultdict
import numpy as np


class FraudDataCollector:
    def __init__(self, host='localhost', port=5555, collection_size=1000):
        """
        Initialize the fraud data collector.
        
        Args:
            host: Server address
                  'localhost' for your own server
                  IP address like '192.168.1.10' for shared server
            port: Server port (usually 5555)
            collection_size: Number of transactions to collect
        """
        self.host = host
        self.port = port
        self.collection_size = collection_size
        self.transactions = []
        self.socket = None
        
        # Statistics tracking
        self.user_stats = defaultdict(lambda: {
            'transaction_count': 0,
            'amounts': [],
            'last_transaction_time': None,
            'fraud_count': 0
        })
        
        print("=" * 70)
        print("FRAUD DATA COLLECTOR")
        print("=" * 70)
        print(f"Server: {self.host}:{self.port}")
        print(f"Target size: {self.collection_size} transactions")
        print("=" * 70)
    
    def connect_to_server(self):
        """Establish connection to the fraud stream server"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        try:
            print(f"\nConnecting to {self.host}:{self.port}...")
            self.socket.connect((self.host, self.port))
            print("✓ Connected successfully!")
            return True
        except ConnectionRefusedError:
            print(f"✗ Error: Could not connect to server at {self.host}:{self.port}")
            print("  Make sure the fraud_stream_server.py is running:")
            print(f"  python3 fraud_stream_server.py --host {self.host} --port {self.port}")
            return False
        except Exception as e:
            print(f"✗ Connection error: {e}")
            return False
    
    def collect_transactions(self):
        """Collect transactions from the stream"""
        if not self.connect_to_server():
            return False
        
        print(f"\nCollecting {self.collection_size} transactions...")
        print("Progress: ", end='', flush=True)
        
        buffer = ""
        progress_interval = max(1, self.collection_size // 50)
        
        try:
            while len(self.transactions) < self.collection_size:
                # Receive data from server
                data = self.socket.recv(4096).decode('utf-8')
                if not data:
                    print("\n✗ Connection closed by server")
                    break
                
                buffer += data
                lines = buffer.split('\n')
                buffer = lines[-1]
                
                # Process complete JSON messages
                for line in lines[:-1]:
                    if line.strip():
                        try:
                            message = json.loads(line)
                            
                            # Skip connection messages
                            if message.get('type') == 'connection':
                                continue
                            
                            # Store transaction
                            self.transactions.append(message)
                            
                            # Update user statistics
                            self._update_user_stats(message)
                            
                            # Show progress
                            if len(self.transactions) % progress_interval == 0:
                                print("█", end='', flush=True)
                            
                        except json.JSONDecodeError as e:
                            print(f"\n⚠ JSON parse error: {e}")
                            continue
            
            print(" ✓")
            return True
            
        except KeyboardInterrupt:
            print("\n\n⚠ Collection interrupted by user")
            return len(self.transactions) > 0
        except Exception as e:
            print(f"\n✗ Error during collection: {e}")
            return False
        finally:
            if self.socket:
                self.socket.close()
    
    def _update_user_stats(self, transaction):
        """Update running statistics for each user"""
        user_id = transaction['userID']
        stats = self.user_stats[user_id]
        
        stats['transaction_count'] += 1
        stats['amounts'].append(transaction['amount'])
        stats['last_transaction_time'] = transaction['timestamp']
        
        if transaction.get('isFraud', False):
            stats['fraud_count'] += 1
    
    def extract_features(self, transaction):
        """
        Extract engineered features from a transaction for ML model.
        
        Returns a dictionary of features that will be used for training.
        """
        user_id = transaction['userID']
        user_stats = self.user_stats[user_id]
        
        # Calculate user-specific features
        user_avg_amount = np.mean(user_stats['amounts']) if user_stats['amounts'] else 0
        user_std_amount = np.std(user_stats['amounts']) if len(user_stats['amounts']) > 1 else 0
        user_max_amount = max(user_stats['amounts']) if user_stats['amounts'] else 0
        
        # Amount deviation from user's normal behavior
        amount_deviation = 0
        if user_avg_amount > 0:
            amount_deviation = (transaction['amount'] - user_avg_amount) / user_avg_amount
        
        # Z-score of amount (how many std devs from user mean)
        amount_zscore = 0
        if user_std_amount > 0:
            amount_zscore = (transaction['amount'] - user_avg_amount) / user_std_amount
        
        # Extract item category from itemID (first character)
        item_category = transaction['itemID'][0] if transaction.get('itemID') else 'U'
        
        # One-hot encode item category
        categories = ['B', 'C', 'D', 'E', 'F']
        category_features = {f'item_category_{cat}': int(item_category == cat) 
                           for cat in categories}
        
        # Extract merchant info
        merchant_id = transaction.get('merchantID', 'M000')
        merchant_num = int(merchant_id[1:]) if len(merchant_id) > 1 else 0
        
        # Time-based features
        hour = transaction['hourOfDay']
        is_night = int(hour >= 22 or hour <= 5)  # 10pm to 5am
        is_business_hours = int(9 <= hour <= 17)
        is_lunch_time = int(11 <= hour <= 13)
        is_evening = int(17 <= hour <= 21)
        
        # Create feature dictionary
        features = {
            # Basic transaction features
            'amount': transaction['amount'],
            'amount_log': np.log1p(transaction['amount']),  # Log transform
            'hour_of_day': hour,
            'is_weekend': int(transaction['isWeekend']),
            'days_since_last_transaction': transaction['daysSinceLastTransaction'],
            
            # Time patterns
            'is_night': is_night,
            'is_business_hours': is_business_hours,
            'is_lunch_time': is_lunch_time,
            'is_evening': is_evening,
            'hour_sin': np.sin(2 * np.pi * hour / 24),  # Circular encoding
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            
            # Item and merchant features
            'merchant_id': merchant_num,
            **category_features,
            
            # User behavior features
            'user_transaction_count': user_stats['transaction_count'],
            'user_avg_amount': user_avg_amount,
            'user_std_amount': user_std_amount,
            'user_max_amount': user_max_amount,
            'amount_deviation': amount_deviation,
            'amount_zscore': amount_zscore,
            'amount_to_user_max_ratio': transaction['amount'] / user_max_amount if user_max_amount > 0 else 0,
            
            # Advanced features
            'is_round_number': int(transaction['amount'] % 100 == 0 or 
                                   transaction['amount'] % 50 == 0),
            'rapid_transaction': int(transaction['daysSinceLastTransaction'] == 0),
            
            # Target variable
            'is_fraud': int(transaction.get('isFraud', False))
        }
        
        # Optional: Add fraud pattern for analysis (not for training)
        if 'fraudPattern' in transaction:
            features['fraud_pattern'] = transaction['fraudPattern']
        else:
            features['fraud_pattern'] = 'none'
        
        # Store original fields for reference
        features['transaction_id'] = transaction['transactionID']
        features['user_id'] = transaction['userID']
        features['user_profile'] = transaction.get('userProfile', 'unknown')
        features['timestamp'] = transaction['timestamp']
        
        return features
    
    def save_dataset(self, filename=None):
        """
        Save collected transactions to CSV file with engineered features.
        
        Args:
            filename: Output filename (default: fraud_data_TIMESTAMP.csv)
        """
        if not self.transactions:
            print("⚠ No transactions to save")
            return None
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fraud_data_{timestamp}.csv"
        
        print(f"\nProcessing {len(self.transactions)} transactions...")
        
        # Extract features from all transactions
        feature_list = []
        for transaction in self.transactions:
            features = self.extract_features(transaction)
            feature_list.append(features)
        
        # Create DataFrame
        df = pd.DataFrame(feature_list)
        
        # Save to CSV
        df.to_csv(filename, index=False)
        
        # Print statistics
        print(f"\n{'=' * 70}")
        print("COLLECTION SUMMARY")
        print('=' * 70)
        print(f"Total transactions collected: {len(self.transactions)}")
        print(f"Fraudulent transactions: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
        print(f"Normal transactions: {(~df['is_fraud'].astype(bool)).sum()} ({(1-df['is_fraud'].mean())*100:.2f}%)")
        print(f"\nUnique users: {df['user_id'].nunique()}")
        print(f"Unique user profiles: {df['user_profile'].nunique()}")
        print(f"\nAmount statistics:")
        print(f"  Mean: ${df['amount'].mean():.2f}")
        print(f"  Median: ${df['amount'].median():.2f}")
        print(f"  Std Dev: ${df['amount'].std():.2f}")
        print(f"  Min: ${df['amount'].min():.2f}")
        print(f"  Max: ${df['amount'].max():.2f}")
        
        print(f"\n📁 Dataset saved to: {filename}")
        print(f"   Total features: {len([col for col in df.columns if col not in ['transaction_id', 'user_id', 'user_profile', 'timestamp', 'fraud_pattern']])}")
        print(f"   Dataset shape: {df.shape}")
        
        # Show fraud patterns distribution
        if 'fraud_pattern' in df.columns:
            fraud_patterns = df[df['is_fraud'] == 1]['fraud_pattern'].value_counts()
            if len(fraud_patterns) > 0:
                print(f"\n🚨 Fraud patterns detected:")
                for pattern, count in fraud_patterns.items():
                    print(f"   {pattern}: {count} instances")
        
        print('=' * 70)
        
        # Save metadata
        metadata = {
            'collection_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'server': f"{self.host}:{self.port}",
            'total_transactions': len(self.transactions),
            'fraud_count': int(df['is_fraud'].sum()),
            'fraud_rate': float(df['is_fraud'].mean()),
            'feature_names': [col for col in df.columns if col not in 
                            ['transaction_id', 'user_id', 'user_profile', 'timestamp', 'fraud_pattern', 'is_fraud']]
        }
        
        metadata_filename = filename.replace('.csv', '_metadata.json')
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"📋 Metadata saved to: {metadata_filename}")
        
        return filename
    
    def run(self, output_filename=None):
        """
        Complete workflow: connect, collect, and save data.
        
        Args:
            output_filename: Optional custom output filename
        
        Returns:
            Filename of saved dataset, or None if failed
        """
        # Collect transactions
        if not self.collect_transactions():
            print("\n✗ Collection failed")
            return None
        
        # Save dataset
        filename = self.save_dataset(output_filename)
        
        return filename


def main():
    parser = argparse.ArgumentParser(
        description='Collect fraud detection training data from stream server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect 1000 transactions from local server
  python fraud_collector.py --size 1000
  
  # Collect 5000 transactions and save to custom file
  python fraud_collector.py --size 5000 --output my_dataset.csv
  
  # Connect to remote server
  python fraud_collector.py --host 192.168.1.10 --port 5555 --size 2000
  
  # Quick test collection (100 transactions)
  python fraud_collector.py --size 100
        """
    )
    
    parser.add_argument('--host', default='localhost',
                       help='Server host address (default: localhost)')
    parser.add_argument('--port', type=int, default=5555,
                       help='Server port (default: 5555)')
    parser.add_argument('--size', type=int, default=1000,
                       help='Number of transactions to collect (default: 1000)')
    parser.add_argument('--output', '-o', dest='output_filename',
                       help='Output CSV filename (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Validate size
    if args.size < 10:
        print("⚠ Warning: Collection size should be at least 10 transactions")
        return 1
    
    # Create collector
    collector = FraudDataCollector(
        host=args.host,
        port=args.port,
        collection_size=args.size
    )
    
    # Run collection
    try:
        filename = collector.run(args.output_filename)
        if filename:
            print(f"\n✓ Success! Dataset ready for training.")
            print(f"\nNext steps:")
            print(f"  1. Examine the data: pandas.read_csv('{filename}')")
            print(f"  2. Train your model using this dataset")
            print(f"  3. Test on real-time stream with fraud_detector.py")
            return 0
        else:
            print("\n✗ Collection failed")
            return 1
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
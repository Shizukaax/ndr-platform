#!/usr/bin/env python3
"""
NDR Platform - Basic Usage Example
This example shows how to load data and run anomaly detection.
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.data_manager import DataManager
from core.model_manager import ModelManager

def basic_anomaly_detection():
    """Basic anomaly detection example."""
    print("🔍 NDR Platform - Basic Example")
    print("=" * 40)
    
    # Initialize components
    data_manager = DataManager()
    model_manager = ModelManager()
    
    # Load sample data
    print("\n📂 Loading sample data...")
    data_file = Path("data/examples/sample_network_data.json")
    
    if not data_file.exists():
        print("❌ Sample data not found. Run 'python scripts/setup.py' first.")
        return
    
    # Load and process data
    data = data_manager.load_json_data(str(data_file))
    print(f"✅ Loaded {len(data)} network records")
    
    # Convert to DataFrame for processing
    df = data_manager.json_to_dataframe(data)
    print(f"📊 DataFrame shape: {df.shape}")
    
    # Run anomaly detection
    print("\n🤖 Running anomaly detection...")
    
    # Use Isolation Forest (default model)
    model = model_manager.get_model('IsolationForest')
    
    # Prepare features (simple example)
    numeric_features = ['src_port', 'dst_port', 'bytes', 'packets']
    features = df[numeric_features].fillna(0)
    
    # Train and predict
    model.fit(features)
    anomaly_scores = model.predict(features)
    
    # Show results
    anomalies = df[anomaly_scores == -1]  # -1 indicates anomaly
    print(f"🚨 Found {len(anomalies)} potential anomalies out of {len(df)} records")
    print(f"📈 Anomaly rate: {len(anomalies)/len(df)*100:.2f}%")
    
    # Display first few anomalies
    if len(anomalies) > 0:
        print("\n🔍 Sample anomalies:")
        for idx, row in anomalies.head(3).iterrows():
            print(f"   • {row['src_ip']}:{row['src_port']} → {row['dst_ip']}:{row['dst_port']}")
            print(f"     Protocol: {row['protocol']}, Bytes: {row['bytes']}")
    
    return anomalies

if __name__ == "__main__":
    try:
        anomalies = basic_anomaly_detection()
        print("\n✅ Example completed successfully!")
        print("\n💡 Next steps:")
        print("   • Run the full platform: streamlit run run.py")
        print("   • Explore more examples in the examples/ folder")
        print("   • Check documentation in docs/")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   • Run: python scripts/setup.py")
        print("   • Install dependencies: pip install -r requirements.txt")

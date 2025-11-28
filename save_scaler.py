"""
Script to load the training data and save the scaler
Run this ONCE before starting your Flask app

"""

import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import os

print("="*80)
print("CREATING SCALER FROM TRAINING DATA")
print("="*80)

# Load the preprocessed training data
try:
    X_train = pd.read_csv('Preprocessed Data/preprocessed_train.csv')
    print(f"\n✓ Loaded training data: {X_train.shape}")
    
    # Create and fit scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    print("✓ Fitted scaler on training data")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("✓ Saved scaler to: models/scaler.pkl")
    
    # Verify it works
    with open('models/scaler.pkl', 'rb') as f:
        test_scaler = pickle.load(f)
    
    print("✓ Verified scaler can be loaded")
    print(f"\nScaler expects {len(test_scaler.feature_names_in_)} features:")
    print(test_scaler.feature_names_in_.tolist())
    
    print("\n" + "="*80)
    print("SUCCESS! Scaler saved and ready to use")
    print("="*80)
    
except FileNotFoundError:
    print("\n❌ Error: preprocessed_train.csv not found!")
    print("Make sure you've run the preprocessing notebook first.")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
import pickle
import pandas as pd

# Load the model
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Get the exact feature names the model expects
if hasattr(model, 'feature_names_in_'):
    model_features = list(model.feature_names_in_)
    print(f"✓ Model expects {len(model_features)} features:\n")
    for i, feature in enumerate(model_features, 1):
        print(f"{i:2d}. {feature}")
    
    # Save to file
    with open('models/required_features.txt', 'w') as f:
        for feature in model_features:
            f.write(f"{feature}\n")
    
    print(f"\n✓ Saved to models/required_features.txt")
    
    # Also save as pickle for easy loading
    with open('models/required_features.pkl', 'wb') as f:
        pickle.dump(model_features, f)
    
    print("✓ Saved to models/required_features.pkl")
else:
    print("⚠ Model doesn't have feature_names_in_ attribute")
    print("Loading from preprocessed data instead...")
    
    # Load from preprocessed data
    X_train = pd.read_csv('Preprocessed Data/preprocessed_train.csv')
    model_features = X_train.columns.tolist()
    
    print(f"\n✓ Found {len(model_features)} features from training data:\n")
    for i, feature in enumerate(model_features, 1):
        print(f"{i:2d}. {feature}")
    
    # Save
    with open('models/required_features.txt', 'w') as f:
        for feature in model_features:
            f.write(f"{feature}\n")
    
    with open('models/required_features.pkl', 'wb') as f:
        pickle.dump(model_features, f)
    
    print(f"\n✓ Saved feature list to models/")
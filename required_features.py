
import pickle
import os

print("="*80)
print("SAVING REQUIRED FEATURES LIST")
print("="*80)

# These are the exact 40 features your model expects (from the scaler output)
REQUIRED_FEATURES = [
    'FORMVER', 'VISITMO', 'VISITDAY', 'VISITYR', 'NACCVNUM', 'NACCAVST', 
    'NACCNVST', 'BIRTHMO', 'BIRTHYR', 'SEX', 'HISPANIC', 'RACE', 'EDUC', 
    'MARISTAT', 'HANDED', 'INBIRMO', 'INBIRYR', 'INSEX', 'INRELTO', 'HEIGHT', 
    'WEIGHT', 'BPSYS', 'HRATE', 'VISION', 'VISCORR', 'VISWCORR', 'HEARING', 
    'HEARAID', 'HEARWAID', 'NACCDIED', 'AGE_AT_VISIT', 'BMI', 'EDUC_LEVEL', 
    'HIGH_BP', 'PACKET_I', 'PACKET_IT', 'PACKET_T', 'AGE_AT_VISIT EDUC', 
    'AGE_AT_VISIT BMI', 'EDUC BMI'
]

try:
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the features list
    with open('models/required_features.pkl', 'wb') as f:
        pickle.dump(REQUIRED_FEATURES, f)
    
    print(f"\n✓ Saved {len(REQUIRED_FEATURES)} required features")
    print("✓ File saved to: models/required_features.pkl")
    
    # Verify it works
    with open('models/required_features.pkl', 'rb') as f:
        loaded_features = pickle.load(f)
    
    print("✓ Verified file can be loaded")
    print(f"\nFirst 5 features: {loaded_features[:5]}")
    print(f"Last 5 features: {loaded_features[-5:]}")
    
    print("\n" + "="*80)
    print("SUCCESS! Required features saved")
    print("="*80)
    print("\nYour Flask app should now work without warnings!")
    print("Restart your app with: python app.py")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
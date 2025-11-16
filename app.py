from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ============================================================================
# LOAD MODEL AND REQUIRED FEATURES
# ============================================================================

try:
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"⚠ Error loading model: {e}")
    model = None

try:
    with open('models/required_features.pkl', 'rb') as f:
        REQUIRED_FEATURES = pickle.load(f)
    print(f"✓ Loaded {len(REQUIRED_FEATURES)} required features")
    print(f"  Features: {REQUIRED_FEATURES[:5]}... (first 5)")
except Exception as e:
    print(f"⚠ Error loading features: {e}")
    REQUIRED_FEATURES = None

print(f"\n{'='*80}")
print("DEMENTIA RISK ESTIMATOR - WEB APPLICATION")
print(f"{'='*80}\n")

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/questionnaire')
def questionnaire():
    return render_template('questionnaire.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        print(f"\n{'='*80}")
        print("NEW PREDICTION REQUEST")
        print(f"{'='*80}")
        
        # Get form data
        form_data = request.get_json()
        print(f"✓ Received {len(form_data)} form responses")
        
        # Create features matching EXACT model requirements
        features_df = create_model_features(form_data)
        print(f"✓ Created feature vector with {len(features_df.columns)} features")
        
        # Debug: Show what we're sending
        print(f"\nFeatures being sent to model:")
        print(f"  Columns: {list(features_df.columns)}")
        print(f"  Sample values: {features_df.iloc[0, :5].to_dict()}")
        
        # Make prediction
        if model is not None:
            probability = model.predict_proba(features_df)[0][1]
            risk_percentage = float(probability * 100)
            prediction = int(model.predict(features_df)[0])
            
            print(f"\n✓ Prediction successful!")
            print(f"  Risk: {risk_percentage:.1f}%")
            print(f"{'='*80}\n")
            
            # Risk category
            if risk_percentage < 30:
                risk_category = "Low Risk"
                risk_color = "#00d97e"
            elif risk_percentage < 70:
                risk_category = "Medium Risk"
                risk_color = "#ffc107"
            else:
                risk_category = "At Risk"
                risk_color = "#dc3545"
            
            # Recommendations
            recommendations = generate_recommendations(form_data)
            
            return jsonify({
                'success': True,
                'risk_percentage': round(risk_percentage, 1),
                'risk_category': risk_category,
                'risk_color': risk_color,
                'prediction': prediction,
                'recommendations': recommendations
            })
        else:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 500
            
    except Exception as e:
        print(f"\n{'='*80}")
        print("❌ PREDICTION ERROR")
        print(f"{'='*80}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'='*80}\n")
        
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/results')
def results():
    return render_template('results.html')

# ============================================================================
# FEATURE ENGINEERING - MATCHES EXACT MODEL TRAINING
# ============================================================================

def create_model_features(form_data):
    """
    Create features matching EXACTLY what the model expects
    Based on required_features.pkl: 40 specific features
    """
    
    # Initialize feature dictionary
    features = {}
    
    # === EXACT FEATURES FROM MODEL (in order) ===
    
    # 1. FORMVER
    features['FORMVER'] = 3
    
    # 2-5. Visit date
    features['VISITMO'] = 11
    features['VISITDAY'] = 17
    features['VISITYR'] = 2024
    
    # 6-7. Visit numbers
    features['NACCVNUM'] = 1
    features['NACCAVST'] = 1
    features['NACCNVST'] = 0
    
    # 8-9. Birth date (estimate from age question)
    features['BIRTHMO'] = 6
    features['BIRTHYR'] = 1965 if form_data.get('born_before_1970') == 'yes' else 1985
    
    # 10-14. Demographics
    features['SEX'] = 1 if form_data.get('sex_male') == 'yes' else 0
    features['HISPANIC'] = 1 if form_data.get('hispanic_latino') == 'yes' else 0
    features['RACE'] = 1 if form_data.get('racial_minority') == 'yes' else 0
    features['EDUC'] = 16 if form_data.get('education_years') == 'yes' else 10
    features['MARISTAT'] = 1 if form_data.get('married_partnered') == 'yes' else 0
    
    # 15. Handedness
    features['HANDED'] = 1 if form_data.get('right_handed') == 'yes' else 0
    
    # 16-19. Informant data
    features['INBIRMO'] = 6
    features['INBIRYR'] = features['BIRTHYR']
    features['INSEX'] = features['SEX']
    features['INRELTO'] = 1
    
    # 20-21. Physical measurements
    features['HEIGHT'] = 170 if form_data.get('height_above_150') == 'yes' else 145
    features['WEIGHT'] = 70 if form_data.get('weight_above_50') == 'yes' else 45
    
    # 22-23. Vital signs
    if form_data.get('systolic_bp') == 'yes':
        features['BPSYS'] = 120
    elif form_data.get('systolic_bp') == 'no':
        features['BPSYS'] = 150
    else:
        features['BPSYS'] = 130
    
    if form_data.get('heart_rate_normal') == 'yes':
        features['HRATE'] = 75
    elif form_data.get('heart_rate_normal') == 'no':
        features['HRATE'] = 105
    else:
        features['HRATE'] = 80
    
    # 24-29. Vision & Hearing
    features['VISION'] = 0 if form_data.get('vision_without_glasses') == 'yes' else 1
    features['VISCORR'] = 0 if form_data.get('vision_with_glasses') == 'yes' else 1
    features['VISWCORR'] = features['VISCORR']
    features['HEARING'] = 0 if form_data.get('hearing_without_aid') == 'yes' else 1
    features['HEARAID'] = 1 if form_data.get('hearing_with_aid') != 'na' else 0
    features['HEARWAID'] = 0 if form_data.get('hearing_with_aid') == 'yes' else 1
    
    # 30. Death indicator
    features['NACCDIED'] = 0
    
    # 31. AGE_AT_VISIT (engineered)
    features['AGE_AT_VISIT'] = features['VISITYR'] - features['BIRTHYR']
    
    # 32. BMI (engineered)
    height_m = features['HEIGHT'] * 0.0254
    weight_kg = features['WEIGHT'] * 0.453592
    features['BMI'] = np.clip(weight_kg / (height_m ** 2), 10, 60)
    
    # 33. EDUC_LEVEL (engineered)
    if features['EDUC'] >= 16:
        features['EDUC_LEVEL'] = 2
    elif features['EDUC'] >= 12:
        features['EDUC_LEVEL'] = 1
    else:
        features['EDUC_LEVEL'] = 0
    
    # 34. HIGH_BP (engineered)
    # Calculate diastolic BP first
    if form_data.get('diastolic_bp') == 'yes':
        diastolic = 80
    elif form_data.get('diastolic_bp') == 'no':
        diastolic = 95
    else:
        diastolic = 85
    
    features['HIGH_BP'] = 1 if (features['BPSYS'] > 140 or diastolic > 90) else 0
    
    # 35-37. PACKET one-hot encoding
    is_remote = form_data.get('remote_response') == 'yes'
    features['PACKET_I'] = 0 if is_remote else 1
    features['PACKET_IT'] = 0  # Default to 0 (not in-person phone)
    features['PACKET_T'] = 1 if is_remote else 0
    
    # 38-40. Polynomial/interaction features
    features['AGE_AT_VISIT EDUC'] = features['AGE_AT_VISIT'] * features['EDUC']
    features['AGE_AT_VISIT BMI'] = features['AGE_AT_VISIT'] * features['BMI']
    features['EDUC BMI'] = features['EDUC'] * features['BMI']
    
    # Convert to DataFrame with EXACT column order from model
    if REQUIRED_FEATURES is not None:
        # Use exact order from model
        df = pd.DataFrame([features])[REQUIRED_FEATURES]
    else:
        # Fallback: create with features in order
        df = pd.DataFrame([features])
    
    # Ensure float32 dtype
    df = df.astype('float32')
    
    return df

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

def generate_recommendations(form_data):
    """Generate personalized recommendations"""
    recommendations = []
    
    if form_data.get('education_years') == 'no':
        recommendations.append({
            'icon': '📚',
            'title': 'Cognitive Engagement',
            'text': 'Engage in mentally stimulating activities like reading, puzzles, or learning new skills.'
        })
    
    if form_data.get('married_partnered') == 'no':
        recommendations.append({
            'icon': '👥',
            'title': 'Social Connection',
            'text': 'Maintain strong social ties through clubs, volunteering, or regular family contact.'
        })
    
    if form_data.get('smoking_history') == 'yes':
        recommendations.append({
            'icon': '🚭',
            'title': 'Quit Smoking',
            'text': 'Smoking cessation can reduce dementia risk at any age. Seek support to quit.'
        })
    
    if form_data.get('hearing_without_aid') == 'no':
        recommendations.append({
            'icon': '👂',
            'title': 'Hearing Health',
            'text': 'Untreated hearing loss is linked to cognitive decline. Get a hearing evaluation.'
        })
    
    if form_data.get('systolic_bp') == 'no' or form_data.get('diastolic_bp') == 'no':
        recommendations.append({
            'icon': '❤️',
            'title': 'Blood Pressure',
            'text': 'High blood pressure increases dementia risk. Monitor and manage with your doctor.'
        })
    
    # Always include
    recommendations.extend([
        {
            'icon': '🥗',
            'title': 'Brain-Healthy Diet',
            'text': 'Follow the MIND diet: vegetables, berries, whole grains, fish, healthy fats.'
        },
        {
            'icon': '🏃',
            'title': 'Physical Activity',
            'text': 'Aim for 150 minutes of moderate exercise weekly for brain health.'
        },
        {
            'icon': '😴',
            'title': 'Quality Sleep',
            'text': 'Prioritize 7-8 hours of quality sleep to clear brain waste products.'
        }
    ])
    
    return recommendations[:6]

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    print("✓ Server starting on http://localhost:5000")
    print(f"{'='*80}\n")
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
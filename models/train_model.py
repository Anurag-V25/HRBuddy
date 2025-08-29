import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import sqlite3
import os

def load_real_data():
    """Load your actual employee data"""
    DB_PATH = 'data/processed/hr_data.db'  # Your database path
    
    try:
        conn = sqlite3.connect(DB_PATH)
        query = """
        SELECT 
            e.*,
            eng.jobsatisfactionscore, 
            eng.worklifebalancerating, 
            c.monthlysalary,
            rs.riskscore
        FROM employees e
        LEFT JOIN engagement eng ON e.employeeid = eng.employeeid
        LEFT JOIN compensation c ON e.employeeid = c.employeeid
        LEFT JOIN risk_scores rs ON e.employeeid = rs.employeeid
        """
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Standardize columns
        df.columns = [col.strip().lower() for col in df.columns]
        
        print(f"✅ Loaded {len(df)} real employee records")
        print(f"✅ Employee ID range: {df['employeeid'].min()} to {df['employeeid'].max()}")
        
        return df
    except Exception as e:
        print(f"❌ Error loading real data: {e}")
        return None

def prepare_features(df):
    """Prepare features for ML model"""
    # Create attrition target (you may have this in your data already)
    if 'attrition' not in df.columns:
        # Create attrition based on risk score and other factors
        df['attrition'] = ((df['riskscore'] > 0.7) & 
                          (df['jobsatisfactionscore'] < 3)).astype(int)
    
    # Select features for model
    feature_cols = ['yearsofexperience', 'jobsatisfactionscore', 'worklifebalancerating', 
                   'monthlysalary', 'performancerating', 'jobrole']
    
    # Keep only available columns
    available_features = [col for col in feature_cols if col in df.columns]
    
    X = df[available_features + ['employeeid']].copy()
    y = df['attrition']
    
    # Encode categorical variables
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        if col != 'employeeid':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    return X, y, label_encoders

def train_attrition_model():
    """Train new attrition prediction model"""
    print("🚀 Training new attrition model...")
    
    # Load real data
    df = load_real_data()
    if df is None:
        print("❌ Cannot train model - no real data available")
        return None
    
    # Prepare features
    X, y, encoders = prepare_features(df)
    employee_ids = X['employeeid']
    X = X.drop('employeeid', axis=1)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"✅ Model AUC Score: {auc:.3f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and encoders
    os.makedirs('models', exist_ok=True)
    model_data = {
        'model': model,
        'feature_names': X.columns.tolist(),
        'encoders': encoders
    }
    
    joblib.dump(model_data, 'models/attrition_pipeline_real.joblib')
    print("✅ Model saved to models/attrition_pipeline_real.joblib")
    
    return model_data

if __name__ == "__main__":
    train_attrition_model()

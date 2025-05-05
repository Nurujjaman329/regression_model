import pickle
from datetime import datetime

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def prepare_data(data):
    df = pd.json_normalize(data['data'])
    features = pd.DataFrame()

    # Customer Features
    features['customer_is_blacklisted'] = df['customer.customer.isBlacklisted']
    features['customer_name_length'] = df['customer.name'].str.len()

    # Order Amount Features
    features['cart_total'] = df['amount.cartTotal']
    features['total_amount'] = df['amount.total']
    features['shipping_charge'] = df['amount.charges'].apply(lambda x: x[0]['amount'] if x else 0)
    features['discount_ratio'] = (df['amount.cartTotal'] - df['amount.total']) / df['amount.cartTotal'].replace(0, 1)

    # Product Features
    features['product_count'] = df['cart'].apply(len)
    features['total_quantity'] = df['cart'].apply(lambda x: sum(item['quantity'] for item in x))
    features['avg_item_price'] = df['amount.cartTotal'] / features['total_quantity']

    # Payment Method
    features['payment_method'] = df['paymentMethod']

    # Location Features
    features['city'] = df['shippingAddress.city']
    features['district'] = df['shippingAddress.district']

    # Time Features
    df['createAt'] = pd.to_datetime(df['createAt'], errors='coerce')
    features['order_hour'] = df['createAt'].dt.hour
    features['is_weekend'] = df['createAt'].dt.dayofweek >= 5
    features['order_month'] = df['createAt'].dt.month

    # Customer History (optional)
    if 'customer.customer.joinDate' in df.columns:
        df['customer.customer.joinDate'] = pd.to_datetime(df['customer.customer.joinDate'], errors='coerce')
        features['customer_tenure_days'] = (pd.to_datetime('now') - df['customer.customer.joinDate']).dt.days

    # Target Variable
    features['target'] = df['status'].apply(lambda x: 1 if str(x).lower() != 'completed' else 0)

    return features


def train_model(data):
    df = prepare_data(data)
    X = df.drop('target', axis=1)
    y = df['target']

    # Define categorical and numerical features
    categorical_features = ['payment_method', 'city', 'district']
    numerical_features = [col for col in X.columns if col not in categorical_features]

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Build model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced'))
    ])

    model.fit(X, y)

    # Save the model
    with open('order_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model


def predict_order(model, input_data):
    df = pd.DataFrame([input_data])
    features = pd.DataFrame()

    # Basic features
    features['customer_is_blacklisted'] = df.get('customer_is_blacklisted', False)
    features['customer_name_length'] = df['customer_name'].str.len()
    features['cart_total'] = df['cart_total']
    features['total_amount'] = df['total_amount']
    features['shipping_charge'] = df.get('shipping_charge', 0)
    features['discount_ratio'] = (df['cart_total'] - df['total_amount']) / df['cart_total'].replace(0, 1)
    features['product_count'] = df['product_count']
    features['total_quantity'] = df['total_quantity']
    features['avg_item_price'] = df['cart_total'] / df['total_quantity']
    features['payment_method'] = df['payment_method']

    # Location
    features['city'] = df.get('city', 'unknown')
    features['district'] = df.get('district', 'unknown')

    # Time features
    df['order_time'] = pd.to_datetime(df.get('order_time', datetime.now()), errors='coerce')
    features['order_hour'] = df['order_time'].dt.hour
    features['is_weekend'] = df['order_time'].dt.dayofweek >= 5
    features['order_month'] = df['order_time'].dt.month

    # Customer tenure
    if 'customer_join_date' in df.columns:
        df['customer_join_date'] = pd.to_datetime(df['customer_join_date'], errors='coerce')
        features['customer_tenure_days'] = (pd.to_datetime('now') - df['customer_join_date']).dt.days

    # Prediction
    prediction = model.predict(features)
    probability = model.predict_proba(features)

    return {
        'prediction': 'Cancel' if prediction[0] == 1 else 'Complete',
        'probability': float(probability[0][prediction[0]]),
        'features_used': list(features.columns)
    }


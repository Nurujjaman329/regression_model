import pickle
from datetime import datetime

import pandas as pd
from flask import Flask, render_template, request

from model import predict_order

app = Flask(__name__)

# Load the trained model
try:
    with open('order_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: Model not found. Run 'python train_model.py' first.")
    exit(1)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_data = {
            'customer_name': request.form.get('customer_name'),
            'customer_is_blacklisted': request.form.get('customer_is_blacklisted') == 'true',
            'cart_total': float(request.form.get('cart_total')),
            'total_amount': float(request.form.get('total_amount')),
            'shipping_charge': float(request.form.get('shipping_charge', 0)),
            'product_count': int(request.form.get('product_count', 1)),
            'total_quantity': int(request.form.get('total_quantity', 1)),
            'payment_method': request.form.get('payment_method', 'cod'),
            'city': request.form.get('city', 'unknown'),
            'district': request.form.get('district', 'unknown'),
            'order_time': request.form.get('order_time', datetime.now()),
            'customer_join_date': request.form.get('customer_join_date', datetime.now())
        }

        result = predict_order(model, input_data)
        return render_template('index.html', result=result, form_data=input_data)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

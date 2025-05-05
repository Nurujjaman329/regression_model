import json

from model import train_model

# Load your dataset
with open('orders.json', 'r') as f:
    data = json.load(f)

# Train and save the model
model = train_model(data)
print("✅ Model trained and saved as 'order_model.pkl'")

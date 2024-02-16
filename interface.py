import gradio as gr
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load your credit card fraud detection dataset (replace with your dataset)
credit_card_df = pd.read_csv('/content/drive/MyDrive/creditcard.csv')

# Assuming your dataset has columns 'Amount' and 'Time'
X = credit_card_df[['Amount', 'Time']]
y = credit_card_df['Class']  # Assuming 'Class' is the target column

# Feature scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)
def predict_credit_card_fraud(amount, time):
    # Create input DataFrame
    input_data = pd.DataFrame({
        'Amount': [amount],
        'Time': [time]
    })
    # Predict using the trained model
    prediction = model.predict(input_data)
    if prediction[0] == 1 :
      return "Fraud"
    else :
      return "Legit"

# Create Gradio interface
iface = gr.Interface(
    fn=predict_credit_card_fraud,
    inputs=["number", "number"],
    outputs="text",
    title="Credit Card Fraud Detector",
    description="Enter transaction details to predict if it's fraudulent or legitimate."
)
#launching gradio
iface.launch(debug=True)

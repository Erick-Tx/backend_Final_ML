from flask import Flask, request, jsonify
import joblib
import pandas as pd
import socket
import gzip
import pickle
from flask_cors import CORS  # Importar CORS

# Crear la aplicación Flask
app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

# Función para cargar el modelo desde un archivo comprimido
def load_compressed_model(file_path):
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)

# Cargar el modelo y el escalador
model = load_compressed_model("voting_clf_soft_important.pkl.gz")
scaler = joblib.load("scaler.pkl")

# Definir todas las columnas numéricas que requiere el escalador
all_features = ['Monthly_Inhand_Salary', 'Num_of_Delayed_Payment',
                'Num_Credit_Inquiries', 'Amount_invested_monthly',
                'Monthly_Balance', 'Credit_History_Age', 'Annual_Income',
                'Age', 'Num_of_Loan', 'Changed_Credit_Limit', 'Outstanding_Debt',
                'Num_Credit_Card', 'Num_Bank_Accounts', 'Interest_Rate',
                'Delay_from_due_date', 'Credit_Utilization_Ratio', 'Total_EMI_per_month']

# Definir las características utilizadas por el modelo
selected_features = ['Outstanding_Debt', 'Interest_Rate']

# Variable global para almacenar el historial de predicciones
prediction_history = []

@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to the Voting Classifier API. Use /predict for predictions.",
        "routes": {
            "/predict": "Send POST requests with JSON data to make predictions",
            "/predictions": "View the history of predictions made"
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos enviados por el cliente
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        # Crear un DataFrame con los datos enviados
        input_data = pd.DataFrame([data])

        # Completar las columnas faltantes con valores predeterminados (ej. 0)
        for col in all_features:
            if col not in input_data.columns:
                input_data[col] = 0

        # Asegurar el orden de las columnas para el escalador
        input_data = input_data[all_features]

        # Escalar todas las características numéricas
        input_data_scaled = scaler.transform(input_data)

        # Seleccionar únicamente las características importantes para el modelo
        input_data_scaled_selected = pd.DataFrame(input_data_scaled, columns=all_features)[selected_features]

    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # Realizar la predicción
    prediction = model.predict(input_data_scaled_selected)[0]
    probabilities = model.predict_proba(input_data_scaled_selected).tolist()[0]

    # Almacenar la predicción en el historial
    prediction_history.append({
        "input": data,
        "prediction": int(prediction),
        "probabilities": probabilities
    })

    # Devolver el resultado
    return jsonify({
        "prediction": int(prediction),
        "probabilities": probabilities
    })

@app.route('/predictions', methods=['GET'])
def get_predictions():
    # Devolver el historial de predicciones realizadas
    return jsonify({"history": prediction_history})

# Iniciar el servidor
#if __name__ == '__main__':
#    app.run(host="127.0.0.1", port=5000, debug=True)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Toma el puerto de la variable de entorno
    app.run(host="0.0.0.0", port=port)

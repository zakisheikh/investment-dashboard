from flask import Flask, render_template, jsonify
from flask_cors import CORS
from dashboard import get_market_data

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    data, model_mse = get_market_data()  # Unpack the returned tuple
    response_data = {
        "AAPL": {
            "Dates": list(data['AAPL'].index.strftime('%Y-%m-%d')),
            "Close": data['AAPL']['Close'].tolist(),
            "Model MSE": model_mse  # Use the unpacked model_mse
        }
    }
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)

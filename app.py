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
    data = get_market_data()
    response_data = {
        "AAPL": {
            "Dates": list(data['AAPL'].index.strftime('%Y-%m-%d')),
            "Close": data['AAPL']['Close'].tolist(),
            "Model MSE": data['model_mse']  # Assuming model_mse is calculated in get_market_data
        }
    }
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)

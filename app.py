from flask import Flask, render_template, jsonify
import dashboard  # Import your dashboard module

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('dashboard.html')  # Render your main dashboard template

@app.route('/api/data')
def get_data():
    data = dashboard.get_market_data()  # Fetch market data and models
    return jsonify(data)  # Return the data as JSON

if __name__ == '__main__':
    app.run(debug=True)

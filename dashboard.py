from dash import Dash
from dashboard import create_dashboard

app = Dash(__name__)

# Layout for the app is generated from create_dashboard()
app.layout = create_dashboard()

if __name__ == '__main__':
    app.run_server(debug=True)

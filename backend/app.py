import os
import io
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from auth import require_auth
from agent import run_agent

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Configuration
CLERK_PUBLISHABLE_KEY = os.getenv("CLERK_PUBLISHABLE_KEY")
CLERK_FRONTEND_API = os.getenv("CLERK_FRONTEND_API", "https://clerk.clerk.com")

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)
# Load the trained models
def load_models():
    try:
        # Assumes running from project root
        base_path = os.getcwd()
        with open(os.path.join(base_path, 'models', 'logistic_regression_model.pkl'), 'rb') as f:
            lr_model = pickle.load(f)
        with open(os.path.join(base_path, 'models', 'random_forest_model.pkl'), 'rb') as f:
            rf_model = pickle.load(f)
        with open(os.path.join(base_path, 'models', 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        return lr_model, rf_model, scaler
    except FileNotFoundError:
        print("Models not found in 'models/' directory.")
        return None, None, None
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None 
@app.route("/")
def index():
    return render_template("landing.html", clerk_publishable_key=CLERK_PUBLISHABLE_KEY)
def get_metrics_data():
    """Helper to load comprehensive metrics from CSV"""
    metrics_data = {
        'lr': {'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1 Score': 0},
        'rf': {'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1 Score': 0},
        'avg_acc': 0
    }
    try:
        metrics_path = os.path.join(os.getcwd(), 'models', 'model_metrics_summary.csv')
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            # Get Test data metrics
            test_metrics = df[df['Dataset'] == 'Test']
            if not test_metrics.empty:
                for model_type, model_key in [('Logistic Regression', 'lr'), ('Random Forest', 'rf')]:
                    row = test_metrics[test_metrics['Model'] == model_type]
                    if not row.empty:
                        metrics_data[model_key] = {
                            'Accuracy': round(float(row.iloc[0]['Accuracy']) * 100, 1),
                            'Precision': round(float(row.iloc[0]['Precision']) * 100, 1),
                            'Recall': round(float(row.iloc[0]['Recall']) * 100, 1),
                            'F1 Score': round(float(row.iloc[0]['F1 Score']) * 100, 1)
                        }           
                metrics_data['avg_acc'] = round((metrics_data['lr']['Accuracy'] + metrics_data['rf']['Accuracy']) / 2, 1)
                metrics_data['avg_precision'] = round((metrics_data['lr']['Precision'] + metrics_data['rf']['Precision']) / 2, 1)
                metrics_data['avg_recall'] = round((metrics_data['lr']['Recall'] + metrics_data['rf']['Recall']) / 2, 1)
                metrics_data['avg_f1'] = round((metrics_data['lr']['F1 Score'] + metrics_data['rf']['F1 Score']) / 2, 1)
    except Exception as e:
        print(f"Error loading metrics: {e}")
    return metrics_data
@app.route("/home")
def home():# In a real app, you would verify the session cookie here
    return render_template("home.html", 
                         clerk_publishable_key=CLERK_PUBLISHABLE_KEY,
                         active_page='home',
                         metrics=get_metrics_data())
@app.route("/single_player")
def single_player():
    return render_template("home.html", active_page='single_player', clerk_publishable_key=CLERK_PUBLISHABLE_KEY, metrics=get_metrics_data())
@app.route("/multiplayer")
def multiplayer():
    return render_template("home.html", active_page='multiplayer', clerk_publishable_key=CLERK_PUBLISHABLE_KEY, metrics=get_metrics_data())
@app.route("/compare")
def compare():
    return render_template("home.html", active_page='compare', clerk_publishable_key=CLERK_PUBLISHABLE_KEY, metrics=get_metrics_data())
@app.route("/profile")
def profile():
    return render_template("home.html", active_page='profile', clerk_publishable_key=CLERK_PUBLISHABLE_KEY, metrics=get_metrics_data())

# ==================== ACTION ROUTES ====================

@app.route('/evaluate/single', methods=['POST'])
def evaluate_single_player():
    """Evaluate a single player"""
    if request.method == 'POST':
        # Get form data
        name = request.form['name']
        goals = float(request.form['goals'])
        matches = float(request.form['matches'])
        assists = float(request.form['assists'])
        pass_accuracy = float(request.form['pass_accuracy'])
        tackles = float(request.form['tackles'])
        saves = float(request.form['saves'])
        # Load models
        lr_model, rf_model, scaler = load_models()
        if lr_model is None or rf_model is None or scaler is None:
            return "Models not found. Please ensure models/ directory exists.", 500
        # Prepare data for prediction
        # Check scaler feature names if possible to ensure order, but assuming same order as training
        features = np.array([[goals, matches, assists, pass_accuracy, tackles, saves]])
        features_scaled = scaler.transform(features)
        # Make predictions
        lr_prob = lr_model.predict_proba(features_scaled)[0][1]
        rf_prob = rf_model.predict_proba(features_scaled)[0][1]
        # Calculate average probability
        avg_prob = (lr_prob + rf_prob) / 2
        # Determine selection - average of both models >= 0.45
        selected = avg_prob >= 0.45
        # Prepare result
        result = {
            'name': name,
            'logistic_regression_probability': round(lr_prob, 4),
            'random_forest_probability': round(rf_prob, 4),
            'average_probability': round(avg_prob, 4),
            'selected': selected
        }
        return render_template('result.html', result=result, is_single=True, clerk_publishable_key=CLERK_PUBLISHABLE_KEY)

@app.route('/evaluate/multiplayer', methods=['POST'])
def evaluate_multiplayer():
    """Evaluate multiple players from CSV"""
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        lr_model, rf_model, scaler = load_models()
        if lr_model is None:
             return "Models not found", 500
        try:
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            for encoding in encodings:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                return jsonify({'error': 'Could not read CSV file.'})
            required_columns = ['Name', 'Goals', 'Matches', 'Assists', 'Pass Accuracy', 'Tackles', 'Saves']
            if not all(col in df.columns for col in required_columns):
                return jsonify({'error': 'CSV file does not have the required columns'})
            features = df[['Goals', 'Matches', 'Assists', 'Pass Accuracy', 'Tackles', 'Saves']].values
            features_scaled = scaler.transform(features)
            lr_probs = lr_model.predict_proba(features_scaled)[:, 1]
            rf_probs = rf_model.predict_proba(features_scaled)[:, 1]
            avg_probs = (lr_probs + rf_probs) / 2
            selected = avg_probs >= 0.45
            results = []
            for i in range(len(df)):
                results.append({
                    'name': df['Name'].iloc[i],
                    'logistic_regression_probability': round(lr_probs[i], 4),
                    'random_forest_probability': round(rf_probs[i], 4),
                    'average_probability': round(avg_probs[i], 4),
                    'selected': selected[i]
                })
            results = sorted(results, key=lambda x: x['average_probability'], reverse=True)
            return render_template('result.html', results=results, is_single=False, clerk_publishable_key=CLERK_PUBLISHABLE_KEY)
        except Exception as e:
            return jsonify({'error': f'Error processing CSV file: {str(e)}'})

@app.route('/compare/players', methods=['POST'])
def compare_players():
    """Compare two players"""
    if request.method == 'POST':
        lr_model, rf_model, scaler = load_models()
        if lr_model is None:
            return "Models not found", 500
        # Helper to get float from form
        def get_float(key):
            return float(request.form.get(key, 0))
        # Get Player 1 data
        p1_name = request.form['player1_name']
        p1_stats = [get_float('player1_goals'), get_float('player1_matches'), get_float('player1_assists'),
                    get_float('player1_pass_accuracy'), get_float('player1_tackles'), get_float('player1_saves')]
        # Get Player 2 data
        p2_name = request.form['player2_name']
        p2_stats = [get_float('player2_goals'), get_float('player2_matches'), get_float('player2_assists'),
                    get_float('player2_pass_accuracy'), get_float('player2_tackles'), get_float('player2_saves')]
        # Predict
        p1_feat = scaler.transform(np.array([p1_stats]))
        p2_feat = scaler.transform(np.array([p2_stats]))
        # Get individual model probabilities
        p1_lr_prob = lr_model.predict_proba(p1_feat)[0][1]
        p1_rf_prob = rf_model.predict_proba(p1_feat)[0][1]
        p1_prob = (p1_lr_prob + p1_rf_prob) / 2
        p2_lr_prob = lr_model.predict_proba(p2_feat)[0][1]
        p2_rf_prob = rf_model.predict_proba(p2_feat)[0][1]
        p2_prob = (p2_lr_prob + p2_rf_prob) / 2
        if p1_prob > p2_prob:
            winner, winner_name, winner_prob = 'player1', p1_name, p1_prob
        elif p2_prob > p1_prob:
            winner, winner_name, winner_prob = 'player2', p2_name, p2_prob
        else:
            winner, winner_name, winner_prob = 'tie', 'Tie', p1_prob   
        probability_diff = abs(p1_prob - p2_prob)
        # Prepare data dicts with all probabilities
        def create_player_data(name, lr_prob, rf_prob, avg_prob, stats):
            return {
                'name': name,
                'logistic_regression_probability': round(lr_prob, 4),
                'random_forest_probability': round(rf_prob, 4),
                'average_probability': round(avg_prob, 4),
                'selected': avg_prob >= 0.45,
                'stats': {
                    'goals': int(stats[0]), 'matches': int(stats[1]), 'assists': int(stats[2]),
                    'pass_accuracy': stats[3], 'tackles': int(stats[4]), 'saves': int(stats[5])
                }
            }
        return render_template('compare_result.html',
                             player1=create_player_data(p1_name, p1_lr_prob, p1_rf_prob, p1_prob, p1_stats),
                             player2=create_player_data(p2_name, p2_lr_prob, p2_rf_prob, p2_prob, p2_stats),
                             winner=winner,
                             winner_name=winner_name,
                             winner_prob=winner_prob,
                             probability_diff=probability_diff,
                             clerk_publishable_key=CLERK_PUBLISHABLE_KEY)

@app.route('/download_template')
def download_template():
    """Download CSV template"""
    template_data = {
        'Name': ['Player1', 'Player2'],
        'Goals': [10, 5],
        'Matches': [20, 15],
        'Assists': [5, 3],
        'Pass Accuracy': [85, 75],
        'Tackles': [12, 8],
        'Saves': [2, 0]
    }
    df = pd.DataFrame(template_data)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    
    return send_file(
        io.BytesIO(buffer.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='player_template.csv'
    )

@app.route("/agent", methods=["POST"])
def agent_endpoint():
    # 🔐 AUTH HERE
    user = require_auth()   
    
    data = request.json
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    message = data.get("message")
    
    # Run the agent logic
    reply = run_agent(user, message)

    return jsonify({
        "user_id": user.get("sub"),
        "reply": reply
    })

if __name__ == "__main__":
    print(f"Starting Flask server on http://localhost:5000")
    if not CLERK_PUBLISHABLE_KEY:
        print("WARNING: CLERK_PUBLISHABLE_KEY is not set in .env")
    app.run(debug=True, port=5000)

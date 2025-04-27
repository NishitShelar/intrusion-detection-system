from flask import Flask, request, jsonify
import pandas as pd
import time
import threading
import pickle
from flask_cors import CORS

# ==================== SETUP ====================

app = Flask(__name__)
CORS(app)  # Allow frontend to connect easily

# 1. Load column names
column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
]

# 2. Load Encoders
with open('model/protocol_encoder.pkl', 'rb') as f:
    protocol_encoder = pickle.load(f)

with open('model/service_encoder.pkl', 'rb') as f:
    service_encoder = pickle.load(f)

with open('model/flag_encoder.pkl', 'rb') as f:
    flag_encoder = pickle.load(f)

# 3. Load trained model
with open('model/ids_randomforest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 4. Load sample data files
normal_df = pd.read_csv('data/normal.csv')
dos_df = pd.read_csv('data/dos.csv')
probe_df = pd.read_csv('data/probe.csv')
r2l_df = pd.read_csv('data/r2l.csv')
u2r_df = pd.read_csv('data/u2r.csv')

# 5. Attack Mode
attack_mode = 'normal'

# 6. Live Data Storage
current_data_row = {}

# ==================== ATTACK MODE CONTROL ====================

@app.route('/set_attack_mode', methods=['POST'])
def set_attack_mode():
    global attack_mode
    data = request.get_json()
    attack_mode = data.get('attack_mode', 'normal')
    return jsonify({'message': f'✅ Attack mode set to {attack_mode}'}), 200

# ==================== DATA STREAMING ====================

def data_streaming_thread():
    global current_data_row

    while True:
        if attack_mode == 'normal':
            sample_row = normal_df.sample(1).to_dict(orient='records')[0]
        elif attack_mode == 'DoS':
            sample_row = dos_df.sample(1).to_dict(orient='records')[0]
        elif attack_mode == 'Probe':
            sample_row = probe_df.sample(1).to_dict(orient='records')[0]
        elif attack_mode == 'R2L':
            sample_row = r2l_df.sample(1).to_dict(orient='records')[0]
        elif attack_mode == 'U2R':
            sample_row = u2r_df.sample(1).to_dict(orient='records')[0]
        else:
            sample_row = normal_df.sample(1).to_dict(orient='records')[0]

        current_data_row = sample_row
        time.sleep(1)

threading.Thread(target=data_streaming_thread, daemon=True).start()

@app.route('/stream_data', methods=['GET'])
def stream_data():
    return jsonify(current_data_row)

# ==================== PREDICTION ====================

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Drop extra fields
        data.pop('attack_category', None)
        data.pop('label', None)

        # Encode categorical fields
        data['protocol_type'] = protocol_encoder.transform([data['protocol_type']])[0]
        data['service'] = service_encoder.transform([data['service']])[0]
        data['flag'] = flag_encoder.transform([data['flag']])[0]

        # Prepare DataFrame
        feature_cols = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
        ]
        df = pd.DataFrame([data], columns=feature_cols)

        # Predict
        prediction = model.predict(df)[0]
        return jsonify({'prediction': prediction}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== MAIN ====================

if __name__ == '__main__':
    app.run(debug=True, port=5000)

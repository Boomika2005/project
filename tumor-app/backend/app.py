# app.py
import io
import mysql.connector
from datetime import datetime
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from report_generator import build_report_pdf
import google.generativeai as genai

app = Flask(__name__)
CORS(app)
model=None
def connect():
    global model
    API_KEY = "AIzaSyCgd_bBl9vHKnU3BUXtYvhhT0pNyf6J6X8"
    genai.configure(api_key=API_KEY)

    # Initialize Gemini model
    model = genai.GenerativeModel("gemini-1.5-flash")
    print('model conneced successfully')

connect()
# -------------------- DATABASE --------------------
DB_CONFIG = {
    "host": "localhost",
    "user": "root",          # change if needed
    "password": "Boomika236",  # change if needed
    "database": "tumor_app"
}

def init_db():
    conn = mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"]
    )
    cursor = conn.cursor()
    cursor.execute("CREATE DATABASE IF NOT EXISTS tumor_app")
    conn.database = "tumor_app"

    # Patients table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INT AUTO_INCREMENT PRIMARY KEY,
            patient_id VARCHAR(50) UNIQUE,
            name VARCHAR(100),
            age INT,
            sex VARCHAR(10),
            password VARCHAR(255)
        )
    ''')

    # Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            patient_id VARCHAR(50),
            prediction_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            report_filename VARCHAR(255),
            result TEXT,
            tumor_type VARCHAR(50),
            tumor_size FLOAT,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
                ON DELETE CASCADE
                ON UPDATE CASCADE
        )
    ''')

    # Feedback table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100),
            email VARCHAR(100),
            message TEXT
        )
    ''')

    conn.commit()
    conn.close()

# init_db()

# -------------------- ROUTES ----------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# -------- Register --------
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    patient_id = data.get("patient_id")
    name = data.get("name")
    age = data.get("age")
    sex = data.get("sex")
    password = data.get("password")

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        query = "INSERT INTO patients (patient_id, name, age, sex, password) VALUES (%s, %s, %s, %s, %s)"
        cursor.execute(query, (patient_id, name, age, sex, password))
        conn.commit()
        conn.close()
        return jsonify({"message": "Registration successful"}), 201
    except mysql.connector.IntegrityError:
        return jsonify({"error": "Patient ID already exists"}), 400




# -------- Login --------
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    patient_id = data.get("patient_id")
    password = data.get("password")

    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    query = "SELECT * FROM patients WHERE patient_id=%s AND password=%s"
    cursor.execute(query, (patient_id, password))
    user = cursor.fetchone()
    conn.close()

    if user:
        return jsonify({"message": "Login successful", "patient_id": patient_id}), 200
    else:
        return jsonify({"error": "Invalid credentials"}), 401


# -------- Prediction --------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file part named 'image'"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # need patient_id from frontend
    patient_id = request.form.get("patient_id")
    if not patient_id:
        return jsonify({"error": "Missing patient_id"}), 400

    image_bytes = file.read()

    try:
        # Fetch patient info from DB
        conn = mysql.connector.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database="tumor_app"
        ) #need to  configure
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT name, age, sex FROM patients WHERE patient_id = %s", (patient_id,))
        patient = cursor.fetchone()
        conn.close()

        if not patient:
            return jsonify({"error": "Patient not found"}), 404

        # Example ML result summary
        result_summary = "Tumor Detected"

        # Report filename with timestamp
        report_filename = f"Brain_Tumor_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        # Generate PDF report with patient details + tumor type/size
        pdf_bytes, metadata = build_report_pdf(
            image_bytes=image_bytes,
            result_summary=result_summary,
            patient_id=patient_id,
            patient_name=patient["name"],
            patient_age=patient["age"],
            patient_sex=patient["sex"]
        )

        tumor_type = metadata["tumor_type"]
        tumor_size = metadata["tumor_size"]

        # Save prediction into DB
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (patient_id, report_filename, result, tumor_type, tumor_size)
            VALUES (%s, %s, %s, %s, %s)
        """, (patient_id, report_filename, result_summary, tumor_type, tumor_size))
        conn.commit()
        conn.close()

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=report_filename,
    )


# -------- Feedback --------
@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    patient_id = data.get("patient_id")
    message = data.get("message")
    print(data,patient_id)
    print('reacher here')

    if not patient_id or not message:
        return jsonify({"error": "Patient ID and message are required"}), 400

    try:
        conn = mysql.connector.connect(    
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database="tumor_app"
        )
        cursor = conn.cursor()

        query = "INSERT INTO feedback (patient_id, message) VALUES (%s, %s)"
        cursor.execute(query, (patient_id, message))
        conn.commit()

        cursor.close()
        conn.close()
        return jsonify({"message": "Feedback submitted successfully!"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/chat", methods=["POST"])
def chat():
    
    global model
    print("message from frontend")
    try:
        data = request.json
        user_message = data.get("message", "")

        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        response = model.generate_content(user_message)
        return jsonify({"reply": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/feedback", methods=["GET"])
def get_feedback():
    conn = mysql.connector.connect(    
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"]
        )
    cursor = conn.cursor(dictionary=True)
    conn.database = "tumor_app"
    cursor.execute("SELECT * FROM feedback")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(rows)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
# app.py
from report_generator import build_report_pdf, build_summary_pdf
import io
import mysql.connector
from datetime import datetime
from flask import Flask, request, send_file, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai
from flask import Flask, request, jsonify, send_file
# import mysql.connector
from mysql.connector import pooling
import zipfile

app = Flask(__name__)
CORS(app)
model=None

DB_CONFIG = {
    "host": "34.93.58.125",
    "user": "root",
    "password": "Boomika123#",
    "database": "tumor_app"
}

try:
    pool = pooling.MySQLConnectionPool(
        pool_name="mypool",
        pool_size=3,
        host=DB_CONFIG["host"], 
        user=DB_CONFIG["user"], 
        password=DB_CONFIG["password"], 
        database=DB_CONFIG["database"]
    )
    print("DB pool created successfully")
except mysql.connector.Error as err:
    print("Error creating DB pool:", err)
    pool = None
def connect():
    global model
    API_KEY = "AIzaSyCgd_bBl9vHKnU3BUXtYvhhT0pNyf6J6X8"
    genai.configure(api_key=API_KEY)

    # Initialize Gemini model
    model = genai.GenerativeModel("gemini-1.5-flash")
    print('model conneced successfully')

connect()
# -------------------- DATABASE --------------------

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


#------------------------all get folder ---------------------


@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")


@app.route("/login", methods=["GET"])
def login():
    return render_template("login.html")


@app.route("/register", methods=["GET"])
def register():
    return render_template("register.html")

@app.route("/report", methods=["GET"])
def report():
    return render_template("report.html")

@app.route("/index", methods=["GET"])
def index():
    return render_template("index.html")


# @app.route("/")
# def home():
#     return "Hello, World!"

# -------- Register --------
@app.route("/register_post", methods=["POST"])
def register_post():
    global pool
    data = request.json
    patient_id = data.get("patient_id")
    name = data.get("name")
    age = data.get("age")
    sex = data.get("sex")
    password = data.get("password")

    try:
        # conn = mysql.connector.connect( 
        #     host=DB_CONFIG["host"],
        #     user=DB_CONFIG["user"],
        #     password=DB_CONFIG["password"],
        #     database=DB_CONFIG['database'])
        conn=pool.get_connection()
        cursor = conn.cursor()

        query = "INSERT INTO patients (patient_id, name, age, sex, password) VALUES (%s, %s, %s, %s, %s)"
        cursor.execute(query, (patient_id, name, age, sex, password))
        conn.commit()
        conn.close()
        return jsonify({"message": "Registration successful"}), 201
    except mysql.connector.IntegrityError:
        return jsonify({"error": "Patient ID already exists"}), 400




# -------- Login --------
@app.route("/login_post", methods=["POST"])
def login_post():
    global pool
    data = request.json
    patient_id = data.get("patient_id")
    password = data.get("password")

    # conn = mysql.connector.connect( 
    #         host=DB_CONFIG["host"],
    #         user=DB_CONFIG["user"],
    #         password=DB_CONFIG["password"],
    #         database=DB_CONFIG['database'])
    conn=pool.get_connection()
    cursor = conn.cursor(dictionary=True)
    query = "SELECT * FROM patients WHERE patient_id=%s AND password=%s"
    cursor.execute(query, (patient_id, password))
    user = cursor.fetchone()
    conn.close()

    if user:
        return jsonify({"message": "Login successful", "patient_id": patient_id}), 200
    else:
        return jsonify({"error": "Invalid credentials"}), 401


# # -------- Prediction --------
# @app.route("/predict", methods=["POST"])
# def predict():
#     if "image" not in request.files:
#         return jsonify({"error": "No file part named 'image'"}), 400

#     file = request.files["image"]
#     if file.filename == "":
#         return jsonify({"error": "No selected file"}), 400

#     # need patient_id from frontend
#     patient_id = request.form.get("patient_id")
#     if not patient_id:
#         return jsonify({"error": "Missing patient_id"}), 400

#     image_bytes = file.read()

#     try:
#         # Fetch patient info from DB
#         conn = mysql.connector.connect(
#             host=DB_CONFIG["host"],
#             user=DB_CONFIG["user"],
#             password=DB_CONFIG["password"],
#             database="tumor_app"
#         ) #need to  configure
#         cursor = conn.cursor(dictionary=True)
#         cursor.execute("SELECT name, age, sex FROM patients WHERE patient_id = %s", (patient_id,))
#         patient = cursor.fetchone()
#         conn.close()

#         if not patient:
#             return jsonify({"error": "Patient not found"}), 404

#         # Example ML result summary
#         result_summary = "Tumor Detected"

#         # Report filename with timestamp
#         report_filename = f"Brain_Tumor_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

#         # Generate PDF report with patient details + tumor type/size
#         pdf_bytes, metadata = build_report_pdf(
#             image_bytes=image_bytes,
#             result_summary=result_summary,
#             patient_id=patient_id,
#             patient_name=patient["name"],
#             patient_age=patient["age"],
#             patient_sex=patient["sex"]
#         )

#         tumor_type = metadata["tumor_type"]
#         tumor_size = metadata["tumor_size"]
#         summary = metadata["gemini_summary"]

#         # Save prediction into DB
#         conn = mysql.connector.connect(**DB_CONFIG)
#         cursor = conn.cursor()
#         cursor.execute("""
#             INSERT INTO predictions (patient_id, report_filename, result, tumor_type, tumor_size)
#             VALUES (%s, %s, %s, %s, %s)
#         """, (patient_id, report_filename, result_summary, tumor_type, tumor_size))
#         conn.commit()
#         conn.close()

#     except Exception as e:
#         return jsonify({"error": f"Processing failed: {str(e)}"}), 500

#     return send_file(
#         io.BytesIO(pdf_bytes),
#         mimetype="application/pdf",
#         as_attachment=True,
#         download_name=report_filename,
#     )



@app.route("/predict_post", methods=["POST"])
def predict_post():
    global poll
    if "image" not in request.files:
        return jsonify({"error": "No file part named 'image'"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    patient_id = request.form.get("patient_id")
    if not patient_id:
        return jsonify({"error": "Missing patient_id"}), 400

    image_bytes = file.read()

    try:
        # ---- Fetch patient info ----
        # conn = mysql.connector.connect(
        #       host=DB_CONFIG["host"],
        #      user=DB_CONFIG["user"],
        #     password=DB_CONFIG["password"],
        #      database="tumor_app"
        # )
        conn=pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT name, age, sex FROM patients WHERE patient_id = %s",
            (patient_id,)
        )
        patient = cursor.fetchone()
        # conn.close()

        if not patient:
            return jsonify({"error": "Patient not found"}), 404

        result_summary = "Tumor Detected"
        report_filename = f"Brain_Tumor_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

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
        summary = metadata["gemini_summary"]

        # Save prediction into DB
        # conn = mysql.connector.connect( 
        #     host=DB_CONFIG["host"],
        #     user=DB_CONFIG["user"],
        #     password=DB_CONFIG["password"],
        #     database=DB_CONFIG['database'])
        # cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO predictions (patient_id, report_filename, result, tumor_type, tumor_size)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (patient_id, report_filename, result_summary, tumor_type, tumor_size)
        )
        conn.commit()
        conn.close()

        # ---- Build ZIP ----
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            zipf.writestr(report_filename, pdf_bytes)
            zipf.writestr("Summary.pdf", build_summary_pdf(metadata))
            zipf.writestr("Summary.txt", summary.encode("utf-8"))

        zip_buffer.seek(0)

        return send_file(
            zip_buffer,
            mimetype="application/zip",
            as_attachment=True,
            download_name="Prediction_Report.zip"
        )

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

# -------- Feedback --------
@app.route("/feedback_post", methods=["POST"])
def feedback_post():
    global pool
    data = request.get_json()
    patient_id = data.get("patient_id")
    message = data.get("message")
    print(data,patient_id)
    print('reacher here')

    if not patient_id or not message:
        return jsonify({"error": "Patient ID and message are required"}), 400

    try:
        # conn = mysql.connector.connect(    
        #     host=DB_CONFIG["host"],
        #     user=DB_CONFIG["user"],
        #     password=DB_CONFIG["password"],
        #     database="tumor_app"
        # )
        conn=pool.get_connection()
        cursor = conn.cursor()

        query = "INSERT INTO feedback (patient_id, message) VALUES (%s, %s)"
        cursor.execute(query, (patient_id, message))
        conn.commit()

        cursor.close()
        conn.close()
        return jsonify({"message": "Feedback submitted successfully!"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/chat_post", methods=["POST"])
def chat_post():
    
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



# @app.route("/feedback", methods=["GET"])
# def get_feedback():
#     conn = mysql.connector.connect(    
#         host=DB_CONFIG["host"],
#         user=DB_CONFIG["user"],
#         password=DB_CONFIG["password"]
#         )
#     cursor = conn.cursor(dictionary=True)
#     conn.database = "tumor_app"
#     cursor.execute("SELECT * FROM feedback")
#     rows = cursor.fetchall()
#     cursor.close()
#     conn.close()
#     return jsonify(rows)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
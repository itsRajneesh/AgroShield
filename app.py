from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, session, send_from_directory
)
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

import os
import json
import random
import numpy as np
import smtplib

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta, timezone

# ==================== ML IMPORTS ====================
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array

# OpenCV for plant leaf detection
import cv2

# ==================== LOAD ENV ====================
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key")

# ==================== MONGODB ====================
app.config["MONGO_URI"] = os.getenv("MONGO_URI")
mongo = PyMongo(app)
users = mongo.db.users

# ==================== TIME ====================
def now_utc():
    return datetime.utcnow()  # Now it Return naive datetime (Without timezone)

# ==================== UPLOAD CONFIG ====================
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ==================== LOAD CLASS NAMES ====================
CLASS_NAMES = []
if os.path.exists("class_names.json"):
    with open("class_names.json", "r") as f:
        CLASS_NAMES = json.load(f)

# ==================== LOAD MODEL (ONCE) ====================
model = None
try:
    # Build the same model architecture
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(len(CLASS_NAMES), activation="softmax")(x)

    model = Model(base_model.input, output)

    # Load weights only
    model.load_weights("plant_disease_model_best.h5")
    print("✅ Model architecture built and weights loaded successfully (TF 2.10 compatible)")
    print("Model output shape:", model.output_shape)
    print("Number of classes:", len(CLASS_NAMES))
except Exception as e:
    print("❌ Model loading failed:", e)

def preprocess_image(img_path):
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

# New function to check if image is a plant leaf using OpenCV (green color dominance)
def is_plant_leaf(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return False

    # Convert to HSV for color filtering
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Green color range for leaves
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Mask for green pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = cv2.countNonZero(mask) / (img.size / 3)

    # If green pixels > 30% of image, and some contours (leaf shape)
    if green_ratio > 0.3:
        # Check for contours to ensure leaf-like shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            return True
    return False

# ==================== EMAIL (OTP) ====================
def send_otp_email(email, otp):
    msg = MIMEMultipart("alternative")
    msg["From"] = os.getenv("EMAIL_USER")
    msg["To"] = email
    msg["Subject"] = "Password Reset OTP"

    html = render_template("otp_email.html", email=email, otp=otp)
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASSWORD"))
        server.sendmail(msg["From"], email, msg.as_string())

# ==================== ROUTES ====================

@app.route("/")
def home():
    return redirect(url_for("login"))

# ---------- LOGIN ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user = users.find_one({"email": email})
        if user and check_password_hash(user["password"], password):
            session["user_id"] = str(user["_id"])
            return redirect(url_for("dashboard"))

        flash("Invalid email or password", "error")

    return render_template("login.html")

# ---------- SIGNUP ----------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"]

        if users.find_one({"email": email}):
            flash("Email already exists", "error")
            return redirect(url_for("signup"))

        users.insert_one({
            "email": email,
            "password": generate_password_hash(request.form["password"]),
            "created_at": now_utc()
        })

        flash("Account created successfully. Please login.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")

# ---------- FORGOT PASSWORD ----------
@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form["email"]
        user = users.find_one({"email": email})

        if not user:
            flash("Email not found", "error")
            return redirect(url_for("forgot_password"))

        otp = str(random.randint(100000, 999999))

        users.update_one(
            {"email": email},
            {"$set": {
                "reset_otp": otp,
                "reset_otp_expiry": now_utc() + timedelta(minutes=10)
            }}
        )

        send_otp_email(email, otp)
        session["reset_email"] = email

        flash("OTP sent to your email", "info")
        return redirect(url_for("reset_password"))

    return render_template("forgot_password.html")

# ---------- RESET PASSWORD ----------
@app.route("/reset_password", methods=["GET", "POST"])
def reset_password():
    email = session.get("reset_email")
    if not email:
        flash("Session expired. Try again.", "error")
        return redirect(url_for("forgot_password"))

    user = users.find_one({"email": email})

    if request.method == "POST":
        otp = request.form["otp"]
        password = request.form["password"]
        confirm = request.form["confirm_password"]

        if password != confirm:
            flash("Passwords do not match", "error")
            return redirect(url_for("reset_password"))

        if otp != user.get("reset_otp"):
            flash("Invalid OTP", "error")
            return redirect(url_for("reset_password"))

        if now_utc() > user.get("reset_otp_expiry"):
            flash("OTP expired", "error")
            return redirect(url_for("forgot_password"))

        users.update_one(
            {"email": email},
            {"$set": {"password": generate_password_hash(password)},
             "$unset": {"reset_otp": "", "reset_otp_expiry": ""}}
        )

        session.pop("reset_email", None)
        flash("Password reset successfully. Please login.", "success")
        return redirect(url_for("login"))

    return render_template("reset_password.html")

# ---------- DASHBOARD ----------
@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")

# ---------- PREDICT ----------
@app.route("/predict", methods=["POST"])
def predict():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if not model:
        flash("ML model not loaded", "error")
        return redirect(url_for("dashboard"))

    file = request.files.get("file")
    if not file or file.filename == "":
        flash("No image selected", "error")
        return redirect(url_for("dashboard"))

    if not allowed_file(file.filename):
        flash("Invalid file type", "error")
        return redirect(url_for("dashboard"))

    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    # First, check if it's a plant leaf using OpenCV
    if not is_plant_leaf(path):
        # label = "Not a Plant Leaf – Please upload a clear plant leaf image"
        label = "🤖 RD-powered system here. Plant ki image daalo, bakchodi nahi."
        confidence_display = "N/A"
    else:
        img = preprocess_image(path)
        preds = model.predict(img)[0]

        idx = np.argmax(preds)
        confidence = float(preds[idx]) * 100

        if confidence < 60:
            label = "No Disease Detected (Low Confidence)"
            confidence_display = round(confidence, 2)
        else:
            label = CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else "Unknown"
            confidence_display = round(confidence, 2)

    return render_template(
        "result.html",
        prediction=label,
        confidence=confidence_display,
        image_url=url_for("uploaded_file", filename=filename)
    )

# ---------- SERVE UPLOAD ----------
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# ---------- LOGOUT ----------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ==================== RUN ====================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  #Using Render PORT variable, default 5000
    app.run(host="0.0.0.0", port=port, debug=False)  # 0.0.0.0 To Connect All debug off for production

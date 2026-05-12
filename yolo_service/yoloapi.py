from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os, shutil, requests, time
import uuid 
import json 
import pika 
from ultralytics import YOLO
from pydantic import BaseModel # <-- ADDED: For defining data models

# --- Prometheus Metrics Imports and Setup (NEW) ---
from prometheus_client import Counter, Gauge, start_http_server
import threading
# ----------------------------------------------------

# --- Firebase Imports and Setup ---
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Initialize Firebase Admin SDK
try:
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not cred_path:
        print(" Firebase credentials not set. CRUD operations will be disabled.")
        db = None
    else:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("Firebase initialized successfully.")

except Exception as e:
    print(f" Failed to initialize Firebase: {e}")
    db = None

# --- New Pydantic Model for LLM Input ---
class LLMAnalysisUpdate(BaseModel):
    """Data model for receiving the LLM analysis text."""
    llm_analysis: str

# model load
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "museum_combined_cpu_train", "weights", "best.pt")

app = FastAPI(title="Artifact Classification API", version="1.0")

try:
    model = YOLO(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load YOLO model at {model_path}: {e}")

# DB service URL
DB_API_URL = os.getenv("DATABASE_URL", "http://database:7000")

# rabbitmq config
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_QUEUE = "yolo_predictions"

# metric definitions 
# Total number of images successfully processed by the API
UPLOADER_IMAGES_TOTAL = Counter(
    'uploader_images_uploaded_total', 
    'Total images uploaded via the API'
) 
# RabbitMQ connection status (1=connected, 0=disconnected)
UPLOADER_RABBITMQ_CONNECTED = Gauge(
    'uploader_rabbitmq_connected', 
    'RabbitMQ connection status (1=connected, 0=disconnected)'
) 

def start_metrics_server(): # <-- NEW
    """Starts the Prometheus metrics server on a separate thread (port 8002)."""
    try:
        start_http_server(8002)
        print("Prometheus metrics server started on port 8002")
    except Exception as e:
        print(f"Failed to start Prometheus server: {e}")

threading.Thread(target=start_metrics_server, daemon=True).start() # start metric server

def publish_to_rabbitmq(message_body):
    """
    Connects to RabbitMQ and publishes the prediction data.
    """
    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=RABBITMQ_HOST)
        )
        channel = connection.channel()

        channel.queue_declare(queue=RABBITMQ_QUEUE, durable=True)

        channel.basic_publish(
            exchange='',
            routing_key=RABBITMQ_QUEUE,
            body=json.dumps(message_body).encode('utf-8'),
            properties=pika.BasicProperties(
                delivery_mode=2  
            )
        )
        print(f"Sent prediction {message_body['prediction_id']} to RabbitMQ")
        connection.close()
        UPLOADER_RABBITMQ_CONNECTED.set(1) # Connection Success

    except Exception as e:
        print(f" Failed to publish to RabbitMQ: {e}")
        UPLOADER_RABBITMQ_CONNECTED.set(0) # Connection Failure

# api end points

@app.post("/predict/")
async def predict_artifact(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    temp_path = os.path.join(base_dir, image.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    try:
        prediction_id = str(uuid.uuid4())

        result = model.predict(temp_path)[0]
        pred_label = result.names[result.probs.top1]

        if "_" in pred_label:
            dept, obj = pred_label.split("_", 1)
            dept = dept.replace("_", " ")
            obj = obj.replace("_", " ")
        else:
            dept, obj = "Unknown", pred_label

        response_data = {
            "prediction_id": prediction_id, 
            "department": dept,
            "object": obj,
            "confidence": float(result.probs.top1conf),
            "timestamp": time.time()
        }
        
        # databse
        try:
            requests.post(
                f"{DB_API_URL}/save/",
                params={
                    "image_name": image.filename,
                    "department": dept,
                    "object_name": obj,
                    "probabilities": str(result.probs.data.tolist()),
                    "prediction_id": prediction_id 
                },
                timeout=5
            )
        except Exception as e:
            print(f"⚠️ Warning: could not save to SQLite DB: {e}")

        # firebase 
        if db:
            try:
                db.collection("predictions").document(prediction_id).set({
                    "prediction_id": prediction_id, 
                    "image_name": image.filename,
                    "department": dept,
                    "object_name": obj,
                    "confidence": response_data["confidence"],
                    "timestamp": firestore.SERVER_TIMESTAMP
                })
            except Exception as e:
                print(f"⚠️ Warning: could not save to Firebase: {e}")

        # publish message to rabbitmq
        message = {
            "prediction_id": prediction_id,
            "object_name": obj,
            "department": dept,
            "image_filename": image.filename
        }
        publish_to_rabbitmq(message)
        
        # update prometheus metrics
        UPLOADER_IMAGES_TOTAL.inc() 

        return JSONResponse(content=response_data)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/history/")
def get_history():
    """Fetch all past predictions from the SQLite database service."""
    try:
        r = requests.get(f"{DB_API_URL}/requests/")
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect to DB: {e}")

@app.get("/history/{prediction_id}")
def get_history_item(prediction_id: str):
    """Fetch a specific prediction from the SQLite database service by its UUID."""
    try:
        r = requests.get(f"{DB_API_URL}/request/{prediction_id}")
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect to DB: {e}")

# crud endpoints

@app.get("/firebase/{doc_id}")
def get_firebase_prediction(doc_id: str):
    if not db:
        raise HTTPException(status_code=503, detail="Firebase service is unavailable.")
    
    doc_ref = db.collection("predictions").document(doc_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        raise HTTPException(status_code=404, detail=f"Prediction with ID '{doc_id}' not found in Firebase.")
    
    result = doc.to_dict()
    result["firebase_id"] = doc.id
    return result

@app.put("/firebase/{doc_id}")
def update_firebase_prediction(doc_id: str, department: str, object_name: str, llm_analysis: str):
    if not db:
        raise HTTPException(status_code=503, detail="Firebase service is unavailable.")

    doc_ref = db.collection("predictions").document(doc_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        raise HTTPException(status_code=404, detail=f"Prediction with ID '{doc_id}' not found.")
        
    try:
        doc_ref.update({
            "department": department,
            "object_name": object_name,
            "llm_analysis": llm_analysis,
            "updated_at": firestore.SERVER_TIMESTAMP
        })
        return {"status": "updated", "firebase_id": doc_id, "new_department": department, "new_object": object_name, "new_llm_analysis": llm_analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update Firebase document: {e}")

@app.delete("/firebase/{doc_id}")
def delete_firebase_prediction(doc_id: str):
    if not db:
        raise HTTPException(status_code=503, detail="Firebase service is unavailable.")
        
    doc_ref = db.collection("predictions").document(doc_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return {"status": "deleted (or not found)", "firebase_id": doc_id} 
        
    try:
        doc_ref.delete()
        return {"status": "deleted", "firebase_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete Firebase document: {e}")

# endpoint for llm ouput to update firebase

@app.put("/firebase/analysis/{prediction_id}")
def update_firebase_llm_analysis(prediction_id: str, analysis: LLMAnalysisUpdate):
    if not db:
        raise HTTPException(status_code=503, detail="Firebase service is unavailable.")
    
    # Use the prediction_id to reference the existing document
    doc_ref = db.collection("predictions").document(prediction_id)
    doc = doc_ref.get()

    if not doc.exists:
        raise HTTPException(status_code=404, detail=f"Prediction with ID '{prediction_id}' not found.")
        
    try:
        doc_ref.update({
            "llm_analysis": analysis.llm_analysis,
            "analysis_updated_at": firestore.SERVER_TIMESTAMP
        })
        return {"status": "updated", "firebase_id": prediction_id, "llm_analysis_length": len(analysis.llm_analysis)}
    except Exception as e:
        # Re-raise as HTTPException with a server error detail
        raise HTTPException(status_code=500, detail=f"Failed to update Firebase document with LLM analysis: {e}")
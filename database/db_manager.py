from fastapi import FastAPI, HTTPException
import sqlite3, json, os
from datetime import datetime
from prometheus_client import Counter, Gauge, start_http_server # <-- NEW
import threading # <-- NEW

app = FastAPI(title="DB Manager API")

DB_PATH = os.path.join(os.path.dirname(__file__), "requests.db")

# --- Prometheus Metrics Setup (NEW) ---
# Total number of database requests (saves or updates)
DB_MESSAGES_TOTAL = Counter(
    'database_messages_total',
    'Total messages (requests) processed by the database service'
) 
# Current number of rows in the main table
DB_CURRENT_DETECTIONS = Gauge(
    'database_current_detections',
    'Number of current detections (rows) in the SQLite DB'
) 

def start_metrics_server(): 
    """Starts the Prometheus metrics server on a separate thread (port 8001)."""
    try:
        start_http_server(8001)
        print("Prometheus metrics server started on port 8001")
    except Exception as e:
        print(f"Failed to start Prometheus server: {e}")

threading.Thread(target=start_metrics_server, daemon=True).start() # <-- Start metrics server in background
# --------------------------------------

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            request_uuid TEXT UNIQUE,
            image_name TEXT,
            department TEXT,
            object_name TEXT,
            probabilities TEXT,
            llm_analysis TEXT DEFAULT NULL -- <-- NEW COLUMN FOR LLM OUTPUT
        )
    """)
    conn.commit()
    conn.close()
    
def get_current_detection_count(): # <-- NEW HELPER
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM requests")
    count = cursor.fetchone()[0]
    conn.close()
    return count

init_db()  # Initialize at startup

@app.post("/save/")
# ADD prediction_id as a parameter
def save_request(image_name: str, department: str, object_name: str, prediction_id: str, probabilities: str = None):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO requests (timestamp, request_uuid, image_name, department, object_name, probabilities)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            prediction_id,  # <-- INSERT THE ID
            image_name,
            department,
            object_name,
            probabilities
        ))
        conn.commit()
        
        # --- Update Prometheus Metrics (NEW) ---
        DB_MESSAGES_TOTAL.inc() # Increment counter for new request
        DB_CURRENT_DETECTIONS.set(get_current_detection_count()) # Update gauge
        # --------------------------------------
        
        conn.close()
        return {"status": "saved", "prediction_id": prediction_id}
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Database save error: {e}")


@app.get("/requests/")
def get_all_requests():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Order by the auto-incrementing id, which is still useful
    cursor.execute("SELECT * FROM requests ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return {"requests": rows}

# UPDATE THIS ENDPOINT to use the new UUID
@app.get("/request/{prediction_id}")
def get_request_by_uuid(prediction_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Query by the new request_uuid column
    cursor.execute("SELECT * FROM requests WHERE request_uuid = ?", (prediction_id,))
    row = cursor.fetchone()
    conn.close()
    return {"request": row}

@app.put("/update_analysis/{prediction_id}")
def update_llm_analysis(prediction_id: str, analysis_text: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Update the llm_analysis column where the unique ID matches
        cursor.execute("""
            UPDATE requests 
            SET llm_analysis = ?
            WHERE request_uuid = ?
        """, (analysis_text, prediction_id))
        
        conn.commit()
        
        # --- Update Prometheus Metrics (NEW) ---
        DB_MESSAGES_TOTAL.inc() 
        # --------------------------------------

    finally:
        conn.close()
    
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail=f"Prediction with ID '{prediction_id}' not found.")
        
    return {"status": "updated", "prediction_id": prediction_id, "source": "SQLite"}

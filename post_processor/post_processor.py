#!/usr/bin/env python3
import pika
import os
import json
import subprocess
import time
import requests
import signal
import sys
from typing import Optional
import traceback
from prometheus_client import Counter, Histogram, Gauge, start_http_server # <-- NEW

# RabbitMQ Config
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_QUEUE = os.getenv("RABBITMQ_QUEUE", "yolo_predictions")
DB_API_URL = os.getenv("DATABASE_URL", "http://database:7000")
YOLO_API_URL = os.getenv("YOLO_API_URL", "http://yoloapi:8000") # <-- ADDED: Define YOLO API URL


# --- BitNet Model Paths (Used by Subprocess) ---
LOCAL_MODEL_PATH = "/post_processor/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
BITNET_INFERENCE_SCRIPT = "/BitNet/run_inference.py" # Path to the official runner
MODEL_DIR = "/post_processor/models/BitNet-b1.58-2B-4T" # Directory path for the runner


_running = True 

# --- Prometheus Metrics Setup ---
# Total number of images successfully processed by the worker
WORKER_IMAGES_TOTAL = Counter(
    'worker_images_processed_total', 
    'Total images processed by BitNet post-processor'
) 
# Histogram of LLM inference time in seconds
WORKER_INFERENCE_TIME = Histogram(
    'worker_inference_seconds', 
    'Time taken for BitNet LLM inference'
) 
# RabbitMQ connection status (1=connected, 0=disconnected)
WORKER_RABBITMQ_CONNECTED = Gauge(
    'worker_rabbitmq_connected', 
    'RabbitMQ connection status (1=connected, 0=disconnected)'
) 

# Start Prometheus metrics server on port 8003
try:
    start_http_server(8003)
    print("Prometheus metrics server started on port 8003")
except Exception as e:
    print(f"Failed to start Prometheus server: {e}")
# --------------------------------------

    
def generate_bitnet_description_via_model(object_name: str, department: str) -> Optional[str]:
    """
    Generates a description for an object by calling the external BitNet inference script
    (run_inference.py) via subprocess, as in-memory loading failed.
    """
    #Define the conversation/prompt structure
    system_prompt = "You are an expert post-processor assistant. Your task is to generate a concise, professional description (2-3 sentences) for a detected object. The description must be suitable for a database entry, focusing on the object's presence and its relevance to the specified department."
    user_prompt = f"Detected Object: '{object_name}' in Department: '{department}'. Generate the description."
    prompt = f"### System:\n{system_prompt}\n\n### User:\n{user_prompt}\n\n### Assistant:\n"
    
    #command array to load and run the mdoel. 
    command = [
        sys.executable,
        BITNET_INFERENCE_SCRIPT,
        "-m", LOCAL_MODEL_PATH,
        "-p", prompt,           
        "-n", "100",            
        "-temp", "0.7",         
    ]

    try:
        print(f"Running LLM inference for object: {object_name} via external BitNet runner.")

        # Execute the subprocess call, capturing stdout and raising an error on failure
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True, 
            encoding='utf-8',
            cwd="/BitNet",
            timeout=45 
        )
        
        generated_text = result.stdout.strip()
        
        # The runner script often echoes the entire prompt and the answer. We must strip the prompt.
        if generated_text.startswith(prompt):
            # Strip the input prompt from the output
            return generated_text[len(prompt):].strip()
            
        return generated_text.strip() # Return full text if prompt stripping fails

    except subprocess.CalledProcessError as e:
        # catch erros from the subprocess 
        print(f"BitNet runner failed with exit code {e.returncode}. Stderr: {e.stderr}")
        return None
    except subprocess.TimeoutExpired:
        print("BitNet runner timed out.")
        return None
    except Exception as e:
        print(f"LLM Inference error: {e}")
        traceback.print_exc(file=sys.stdout)
        return None
    
# --- Fallback generator ---
def generate_bitnet_description_fallback(object_name: str, department: str) -> str:
    """
    Simple deterministic fallback LLM-style description generator used when a real LLM is unavailable.
    """
    return (
        f"The {object_name} located in the {department} department appears to be a culturally significant artifact. "
        f"It likely reflects the artistic techniques and social context of the time when it was produced. "
        f"Materials, design motifs, and any inscriptions would help to date and attribute the item, "
        f"while further technical analysis (e.g., material composition testing) could confirm its provenance."
    )

def generate_bitnet_description(object_name: str, department: str) -> str:
    """
    Factory method that attempts to generate an LLM description using the preferred method:
      1) BitNet model via external runner (primary)
      2) fallback generator (safe and fast)
    """
    # Attempt to use the BitNet model via external runner
    model_result = generate_bitnet_description_via_model(object_name, department)
    
    if model_result:
        return model_result
    else:
        print("Falling back to built-in generator after LLM inference failed.")

    # Default fallback
    return generate_bitnet_description_fallback(object_name, department)

# --- Persistence helpers ---
def save_analysis_to_sqlite(prediction_id: str, analysis_text: str):
    """Sends a PUT request to the database service to update the SQLite record."""
    try:
        response = requests.put(
            f"{DB_API_URL}/update_analysis/{prediction_id}",
            params={"analysis_text": analysis_text},
            timeout=5
        )
        response.raise_for_status()
        print(f"LLM analysis saved to SQLite DB for ID: {prediction_id}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to save LLM analysis to SQLite DB: {e}")

def update_firebase_analysis(prediction_id: str, analysis_text: str):
    """Updates the Firebase document via the YOLO API with the LLM analysis."""
    if YOLO_API_URL: # Use the globally defined constant
        try:
            # Target the new endpoint in yoloapi.py
            url = f"{YOLO_API_URL}/firebase/analysis/{prediction_id}"
            
            # Payload must match the LLMAnalysisUpdate model in yoloapi.py (key is llm_analysis)
            payload = {"llm_analysis": analysis_text} 

            resp = requests.put(
                url,
                json=payload,
                timeout=5
            )
            resp.raise_for_status()
            print(f"Firebase analysis updated via YOLO API for ID: {prediction_id}. Response: {resp.json()}")
            return
        except requests.RequestException as e:
            print(f"Firebase update via YOLO API failed: {e}")
    print("Firebase update logic not configured (YOLO_API_URL is missing or incorrectly set).")


def post_process_data(data):
    prediction_id = data.get('prediction_id')
    object_name = data.get('object_name', 'unknown object')
    department = data.get('department', 'unknown department')

    print(f"Processing Request {prediction_id}: {object_name} in {department}")

    # CALL BITNET / GENERATOR
    start_time = time.time() # <-- sStart timing
    analysis_result = generate_bitnet_description(object_name, department)
    end_time = time.time()
    duration = end_time - start_time
    
    # --- Update Prometheus Metrics  ---
    WORKER_INFERENCE_TIME.observe(duration) # Observe the duration
    # ---------------------------------------

    print(f"BitNet/Fallback Output ({duration:.2f}s): {analysis_result}")

    # PERSISTENCE: Save result(s)
    if prediction_id:
        save_analysis_to_sqlite(prediction_id, analysis_result)
        update_firebase_analysis(prediction_id, analysis_result)
        
        # --- Update Prometheus Metrics  ---
        WORKER_IMAGES_TOTAL.inc() # Increment counter on successful processing
        # ---------------------------------------
    else:
        print("No prediction_id present in message; skipping persistence.")

    print("--- Post-Processing Complete ---")

# --- RabbitMQ consumer logic ---
def callback(ch, method, properties, body):
    try:
        data = json.loads(body.decode('utf-8'))
        post_process_data(data)
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        print(f"Error processing message: {e}")
        try:
            ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
        except Exception:
            pass

def start_consumer():
    global _running
    print("Connecting to RabbitMQ...")

    attempt = 0
    connection = None
    while _running:
        attempt += 1
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
            channel = connection.channel()
            channel.queue_declare(queue=RABBITMQ_QUEUE, durable=True)
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue=RABBITMQ_QUEUE, on_message_callback=callback)
            print("Waiting for messages. To exit press CTRL+C")
            
            WORKER_RABBITMQ_CONNECTED.set(1) # Connection Success
            
            while _running:
                connection.process_data_events(time_limit=1)
            print("Stop requested, closing connection...")
            try:
                connection.close()
            except Exception:
                pass
            break
        except pika.exceptions.AMQPConnectionError:
            sleep_for = min(5 + attempt, 30)
            print(f"RabbitMQ not ready (attempt {attempt}), retrying in {sleep_for}s...")
            WORKER_RABBITMQ_CONNECTED.set(0) # Connection Failure
            time.sleep(sleep_for)
        except Exception as e:
            print(f"Unexpected error while connecting to RabbitMQ: {e}")
            WORKER_RABBITMQ_CONNECTED.set(0) # Connection Failure
            time.sleep(5)


if __name__ == '__main__':
    # Initialize globals for clean shutdown logic
    def signal_handler(sig, frame):
        global _running
        print('\n Signal received, shutting down gracefully...')
        _running = False
        sys.exit(0)

    # Register the signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    start_consumer()
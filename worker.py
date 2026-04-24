import os
import time
import json
import base64
import redis
import logging
from datetime import datetime
import cv2
import numpy as np

# Import your webtest functions
from webtest import (
    CNICProcessor,
    detect_cnic_fields,
    process_cnic_front,
    detect_face_in_image,
    verify_face_live,
    extract_picture_from_cnic,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== CONFIGURATION =====
REDIS_URL = os.getenv("REDIS_URL")
REDIS_QUEUE = "cnic_tasks"

if not REDIS_URL:
    logger.error("❌ REDIS_URL environment variable not set!")
    exit(1)

# ===== REDIS CONNECTION =====
try:
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    logger.info(f"✅ Connected to Redis")
except Exception as e:
    logger.error(f"❌ Failed to connect to Redis: {e}")
    exit(1)

# ===== LOAD CNIC PROCESSOR =====
logger.info("Loading CNIC Processor (this may take a moment)...")
try:
    cnic_processor = CNICProcessor('runs/detect/train3/weights/best.pt')
    logger.info("✅ CNIC Processor loaded successfully!")
except Exception as e:
    logger.warning(f"⚠️ Custom model failed: {e}")
    logger.info("Using default YOLO model...")
    cnic_processor = CNICProcessor('yolov8n.pt')
    cnic_processor.class_names = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
        4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
        8: 'boat', 9: 'traffic light',
    }

# ===== HELPER FUNCTIONS =====
def decode_base64_image(base64_str: str) -> np.ndarray:
    """Convert base64 to OpenCV image with error handling"""
    try:
        if not base64_str:
            logger.error("Empty base64 string provided")
            return None
        
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        image_bytes = base64.b64decode(base64_str)
        
        if not image_bytes:
            logger.error("Decoded bytes are empty")
            return None
        
        image_array = np.frombuffer(image_bytes, np.uint8)
        
        if len(image_array) == 0:
            logger.error("Image array is empty")
            return None
        
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error("Failed to decode image - invalid image format")
            return None
        
        return image
        
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        return None

def store_task_result(task_id: str, result: dict):
    """Store task result directly in Redis with proper type conversion"""
    try:
        logger.info(f"🔍 store_task_result called for {task_id}")
        
        # Convert numpy types to Python native types
        def convert_to_serializable(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        # Convert result dict
        serializable_result = convert_to_serializable(result)
        
        task_data = {
            "task_id": task_id,
            "status": "completed",
            "result": serializable_result,
            "completed_at": datetime.now().isoformat()
        }
        
        # Test JSON serialization
        test_json = json.dumps(task_data)
        logger.info(f"🔍 JSON serialization successful, length: {len(test_json)}")
        
        # Store in Redis
        redis_client.setex(f"task:{task_id}", 3600, test_json)
        logger.info(f"✅ Task {task_id} result stored in Redis")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to store result: {e}", exc_info=True)
        return False

def store_task_error(task_id: str, error_msg: str):
    """Store task error in Redis"""
    try:
        task_data = {
            "task_id": task_id,
            "status": "failed",
            "error": error_msg,
            "completed_at": datetime.now().isoformat()
        }
        redis_client.setex(f"task:{task_id}", 3600, json.dumps(task_data))
        logger.info(f"✅ Task {task_id} error stored in Redis")
    except Exception as e:
        logger.error(f"❌ Failed to store error: {e}")

def process_extract_cnic(task_data: dict):
    """Process CNIC extraction task"""
    task_id = task_data["task_id"]
    
    try:
        image_base64 = task_data.get("image_base64")
        
        if not image_base64:
            error_msg = "No image data in task"
            store_task_error(task_id, error_msg)
            return {"error": error_msg}
        
        image = decode_base64_image(image_base64)
        
        if image is None:
            error_msg = "Failed to decode image"
            store_task_error(task_id, error_msg)
            return {"error": error_msg}
        
        # Process CNIC
        detections = detect_cnic_fields(
            image, cnic_processor.model, cnic_processor.class_names
        )
        
        if not detections:
            error_msg = "No CNIC fields detected in image"
            store_task_error(task_id, error_msg)
            return {"error": error_msg}
        
        extracted_data, _, cnic_picture = process_cnic_front(image, cnic_processor)
        
        # Convert face to base64 if present
        cnic_face_base64 = None
        if cnic_picture is not None:
            _, buffer = cv2.imencode('.jpg', cnic_picture)
            cnic_face_base64 = base64.b64encode(buffer).decode('utf-8')
        
        result = {
            "fields": extracted_data,
            "has_cnic_face": cnic_picture is not None,
            "cnic_face_base64": cnic_face_base64
        }
        
        store_task_result(task_id, result)
        return result
        
    except Exception as e:
        logger.error(f"Error in extract_cnic: {e}", exc_info=True)
        store_task_error(task_id, str(e))
        return {"error": str(e)}

def process_verify_face(task_data: dict):
    """Process face verification task"""
    task_id = task_data["task_id"]
    
    try:
        cnic_base64 = task_data.get("cnic_base64")
        selfie_base64 = task_data.get("selfie_base64")
        
        if not cnic_base64 or not selfie_base64:
            error_msg = "Missing image data in task"
            store_task_error(task_id, error_msg)
            return {"error": error_msg}
        
        cnic_image = decode_base64_image(cnic_base64)
        selfie_image = decode_base64_image(selfie_base64)
        
        if cnic_image is None:
            error_msg = "Failed to decode CNIC image"
            store_task_error(task_id, error_msg)
            return {"error": error_msg}
        
        if selfie_image is None:
            error_msg = "Failed to decode selfie image"
            store_task_error(task_id, error_msg)
            return {"error": error_msg}
        
        logger.info(f"Images decoded successfully")
        
        # Extract face from CNIC
        detections = detect_cnic_fields(
            cnic_image, cnic_processor.model, cnic_processor.class_names
        )
        
        if not detections:
            error_msg = "No CNIC fields detected in image"
            store_task_error(task_id, error_msg)
            return {"error": error_msg}
        
        cnic_picture, _ = extract_picture_from_cnic(cnic_image, detections)
        if cnic_picture is None:
            error_msg = "Could not extract face from CNIC card"
            store_task_error(task_id, error_msg)
            return {"error": error_msg}
        
        # Detect face in selfie
        detected_face_result = detect_face_in_image(selfie_image)
        
        if detected_face_result is None:
            error_msg = "No face detected in the selfie image. Please ensure the image is clear and contains a face."
            store_task_error(task_id, error_msg)
            return {"error": error_msg}
        
        live_face, face_coords = detected_face_result
        
        if live_face is None:
            error_msg = "Face detection failed - could not extract face region"
            store_task_error(task_id, error_msg)
            return {"error": error_msg}
        
        # Verify faces
        verification = verify_face_live(cnic_picture, live_face)
        
        store_task_result(task_id, verification)
        return verification
        
    except Exception as e:
        logger.error(f"Error in verify_face: {e}", exc_info=True)
        store_task_error(task_id, str(e))
        return {"error": str(e)}

# ===== MAIN WORKER LOOP =====
def main():
    logger.info(f"🚀 Worker started. Listening on Redis queue: {REDIS_QUEUE}")
    
    while True:
        try:
            result = redis_client.brpop(REDIS_QUEUE, timeout=5)
            
            if result:
                _, task_json = result
                task_data = json.loads(task_json)
                
                task_id = task_data["task_id"]
                task_type = task_data["type"]
                
                logger.info(f"📋 Processing task {task_id} of type {task_type}")
                
                if task_type == "extract_cnic":
                    process_extract_cnic(task_data)
                elif task_type == "verify_face":
                    process_verify_face(task_data)
                else:
                    logger.warning(f"Unknown task type: {task_type}")
                    store_task_error(task_id, f"Unknown task type: {task_type}")
                
                logger.info(f"✅ Task {task_id} completed")
            
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Redis connection error: {e}")
            time.sleep(5)
        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)
            time.sleep(5)

if __name__ == "__main__":
    main()
from ultralytics import YOLO
import cv2
import easyocr
import csv
import os
import pandas as pd
import re
import json
from datetime import datetime
import numpy as np
import time
import arabic_reshaper
from bidi.algorithm import get_display

# ===== FACE RECOGNITION SETUP =====
FACE_RECOGNITION_AVAILABLE = False
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    print("✅ face_recognition library loaded successfully")
except ImportError as e:
    print(f"⚠️ face_recognition not available: {e}")

# DeepFace is disabled
DEEPFACE_AVAILABLE = False

ARABIC_RANGE_PATTERN = re.compile(r'[\u0600-\u06FF]')

def make_bidi_readable(text: str) -> str:
    """Convert Arabic/Urdu OCR text into visually correct right-to-left form."""
    if not text or not ARABIC_RANGE_PATTERN.search(text):
        return text
    try:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception:
        return text

class CNICProcessor:
    def __init__(self, model_path='runs/detect/train3/weights/best.pt'):
        self.model = YOLO(model_path)
        self.reader = easyocr.Reader(['en'], gpu=False)
        
        self.class_names = {
            0: 'CNIC-HHMI', 1: 'bdate', 2: 'country', 3: 'edate', 
            4: 'fname', 5: 'gender', 6: 'id', 7: 'idate', 
            8: 'name', 9: 'picture'
        }

def detect_cnic_fields(image, model, class_names):
    """Detect CNIC fields using custom YOLO model"""
    results = model(image)
    detections = []
    
    for result in results:
        if result.boxes is not None:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                detections.append({
                    'class_id': cls,
                    'class_name': class_names[cls],
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2]
                })
    
    return detections

def preprocess_image_for_ocr(roi):
    """Enhance image for better OCR results"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    return processed

def clean_extracted_text(text, field_type):
    """Clean and extract relevant text based on field type"""
    if not text:
        return ""
    
    if field_type == 'id':
        cnic_pattern = r'\d{5}-\d{7}-\d{1}'
        match = re.search(cnic_pattern, text)
        if match:
            return match.group(0)
        cnic_pattern_alt = r'\d{5}\s*-\s*\d{7}\s*-\s*\d{1}'
        match = re.search(cnic_pattern_alt, text)
        if match:
            return match.group(0).replace(' ', '')
        return ""
    
    elif field_type in ['bdate', 'idate', 'edate']:
        date_patterns = [
            r'\d{2}\.\d{2}\.\d{4}',
            r'\d{2}-\d{2}-\d{4}',
            r'\d{2}/\d{2}/\d{4}',
            r'\d{1,2}\.\d{1,2}\.\d{4}',
            r'\d{1,2},\d{1,2}\.\d{4}',
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                date_str = match.group(0)
                date_str = date_str.replace(',', '.')
                return date_str
        return ""
    
    elif field_type == 'gender':
        gender_pattern = r'\b(M|F|Male|Female|MALE|FEMALE)\b'
        match = re.search(gender_pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).upper() if len(match.group(0)) == 1 else match.group(0).title()
        
        single_letter = re.search(r'\b([MF])\b', text, re.IGNORECASE)
        if single_letter:
            return single_letter.group(0).upper()
        
        if len(text) < 10:
            if 'M' in text.upper() and 'F' not in text.upper():
                return 'M'
            elif 'F' in text.upper():
                return 'F'
        
        return ""
    
    elif field_type == 'country':
        country_pattern = r'\b(Pakistan|PAKISTAN|PK|UAE|United Arab Emirates)\b'
        match = re.search(country_pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).title()
        return ""
    
    elif field_type in ['name', 'fname']:
        name_phrases_to_remove = [
            "PAKISTAN", "National", "Identity", "Card", "Name", "Father", 
            "ather", "Name:", "Father Name", "Father's name", "Fathers name",
            "Gender", "Country", "Stay", "Identity Number", "Date", "Birth",
            "Issue", "Expiry", "Signature", "Holder", "ISLAMIC", "REPUBLIC",
            "OF", "PAKISTAN", "National Identity Card", "M", "F", "Male", "Female"
        ]
        
        cleaned = text
        for phrase in name_phrases_to_remove:
            pattern = r'\b' + re.escape(phrase) + r'\b'
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        
        cleaned = re.sub(r'[^a-zA-Z\s]', '', cleaned)
        cleaned = ' '.join(cleaned.split())
        words = [w for w in cleaned.split() if len(w) > 1]
        words = [w for w in words if len(w) >= 2]
        
        deduped_words = []
        prev = None
        for w in words:
            if prev is None or w.lower() != prev.lower():
                deduped_words.append(w)
            prev = w
        words = deduped_words
        
        if len(words) > 4:
            words = words[:4]
        
        result = ' '.join(words).strip()
        return make_bidi_readable(result) if result else ""
    
    elif field_type == 'CNIC-HHMI':
        return ""
    
    common_phrases = [
        "PAKISTAN", "National Identity Card", "Name", "Father Name", 
        "Gender", "Country of Stay", "Identity Number", "Date of Issue", 
        "Date of Expiry", "Date of Birth", "Signature", "Holder",
        "ISLAMIC REPUBLIC OF PAKISTAN", "Date", "of", "Birth",
        "Issue", "Expiry", "Identity", "Number", "Country", "Stay"
    ]
    
    cleaned = text
    for phrase in common_phrases:
        cleaned = re.sub(re.escape(phrase), "", cleaned, flags=re.IGNORECASE)
    
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = re.sub(r'^[^\w]+|[^\w]+$', '', cleaned)
    
    return make_bidi_readable(cleaned)

def extract_text_from_roi(image, bbox, reader, field_type=''):
    """Extract text from detected region with improved preprocessing"""
    x1, y1, x2, y2 = map(int, bbox)
    
    padding = 10
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)
    
    roi = image[y1:y2, x1:x2]
    
    if roi.size == 0:
        return ""
    
    processed_roi = preprocess_image_for_ocr(roi)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    
    all_texts = []
    
    for img in [processed_roi, gray_roi]:
        try:
            results = reader.readtext(img, detail=1, paragraph=False, text_threshold=0.3, width_ths=0.7, height_ths=0.7)
            for (_, text, conf) in results:
                if conf > 0.4:
                    all_texts.append((text.strip(), conf))
        except Exception as e:
            continue
    
    if not all_texts:
        return ""
    
    all_texts.sort(key=lambda x: x[1], reverse=True)
    
    best_text = all_texts[0][0]
    cleaned_best = clean_extracted_text(best_text, field_type)
    if cleaned_best:
        return cleaned_best
    
    top_n = min(3, len(all_texts))
    combined_text = ' '.join([text for text, _ in all_texts[:top_n]])
    cleaned_combined = clean_extracted_text(combined_text, field_type)
    
    return cleaned_combined

def extract_picture_from_cnic(image, detections):
    """Extract the picture/face region from CNIC card"""
    picture_regions = []
    
    for detection in detections:
        if detection['class_name'] == 'picture':
            x1, y1, x2, y2 = map(int, detection['bbox'])
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)
            
            picture_roi = image[y1:y2, x1:x2]
            if picture_roi.size > 0:
                picture_regions.append({
                    'image': picture_roi,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': detection['confidence']
                })
    
    if picture_regions:
        best_picture = max(picture_regions, key=lambda x: x['confidence'])
        return best_picture['image'], best_picture['bbox']
    return None, None

def detect_face_in_image(image):
    """Detect face in image using OpenCV Haar Cascade"""
    if image is None:
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        face_roi = image[y:y+h, x:x+w]
        return face_roi, (x, y, w, h)
    return None, None

def preprocess_face_for_recognition(face_image):
    """Preprocess face image for better face_recognition results"""
    # Resize to consistent size
    face = cv2.resize(face_image, (150, 150))
    
    # Apply slight Gaussian blur to reduce noise
    face = cv2.GaussianBlur(face, (3, 3), 0)
    
    # Enhance contrast
    if len(face.shape) == 3:
        lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        lab = cv2.merge([l, a, b])
        face = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return face

def compare_faces_face_recognition(face1, face2):
    """Compare two faces using face_recognition library (improved)"""
    if not FACE_RECOGNITION_AVAILABLE:
        return None, "face_recognition library not available"
    
    try:
        # Preprocess faces for better recognition
        face1_processed = preprocess_face_for_recognition(face1)
        face2_processed = preprocess_face_for_recognition(face2)
        
        # Convert BGR to RGB (face_recognition uses RGB)
        face1_rgb = cv2.cvtColor(face1_processed, cv2.COLOR_BGR2RGB)
        face2_rgb = cv2.cvtColor(face2_processed, cv2.COLOR_BGR2RGB)
        
        # Get face encodings
        encoding1 = face_recognition.face_encodings(face1_rgb)
        encoding2 = face_recognition.face_encodings(face2_rgb)
        
        if len(encoding1) == 0:
            return None, "No face found in CNIC picture"
        if len(encoding2) == 0:
            return None, "No face found in live capture"
        
        # Calculate distance (lower = more similar)
        distance = face_recognition.face_distance([encoding1[0]], encoding2[0])[0]
        
        # Threshold: 0.6 is standard, adjust as needed
        # Lower threshold = stricter matching
        THRESHOLD = 0.6
        is_match = distance < THRESHOLD
        
        # Convert distance to percentage (0% = completely different, 100% = identical)
        # Distance of 0 = 100% match, Distance of 1 = 0% match
        similarity = (1 - min(distance, 1)) * 100
        
        print(f"   Face distance: {distance:.4f}")
        print(f"   Match threshold: {THRESHOLD}")
        print(f"   Similarity score: {similarity:.2f}%")
        
        return {
            'is_match': is_match,
            'similarity': similarity,
            'distance': distance,
            'threshold': THRESHOLD,
            'method': 'face_recognition'
        }, None
        
    except Exception as e:
        return None, f"Error in face_recognition: {str(e)}"

def compare_faces_opencv(face1, face2):
    """Compare faces using OpenCV (histogram comparison) - fallback method"""
    try:
        face1_resized = cv2.resize(face1, (128, 128))
        face2_resized = cv2.resize(face2, (128, 128))
        
        gray1 = cv2.cvtColor(face1_resized, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(face2_resized, cv2.COLOR_BGR2GRAY)
        
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        similarity = correlation * 100
        
        is_match = correlation > 0.7
        
        return {
            'is_match': is_match,
            'similarity': similarity,
            'correlation': correlation,
            'method': 'opencv_histogram'
        }, None
    except Exception as e:
        return None, f"Error in OpenCV comparison: {str(e)}"

def verify_face_live(cnic_picture, live_face):
    """Verify if the face in CNIC picture matches live captured face"""
    print("\n" + "="*60)
    print("LIVE FACE VERIFICATION")
    print("="*60)
    
    if cnic_picture is None:
        return {'error': 'No picture found in CNIC card'}
    
    if live_face is None:
        return {'error': 'No face detected in live capture'}
    
    print(f"CNIC face shape: {cnic_picture.shape}")
    print(f"Live face shape: {live_face.shape}")
    print("🔍 Comparing CNIC face with live capture...")
    
    verification_result = {
        'cnic_face_detected': True,
        'live_face_detected': True,
        'methods_tried': []
    }
    
    # PRIMARY METHOD: face_recognition (best quality)
    if FACE_RECOGNITION_AVAILABLE:
        print("\n📊 Comparing faces using face_recognition library...")
        result, error = compare_faces_face_recognition(cnic_picture, live_face)
        if result:
            verification_result['face_recognition'] = result
            verification_result['methods_tried'].append('face_recognition')
            print(f"   Similarity: {result['similarity']:.2f}%")
            print(f"   Match: {'✅ YES' if result['is_match'] else '❌ NO'}")
            
            # Use face_recognition as the final decision
            verification_result['final_verification'] = result['is_match']
            verification_result['confidence'] = result['similarity']
            verification_result['distance'] = result['distance']
        else:
            print(f"   ⚠️ face_recognition error: {error}")
    
    # FALLBACK METHOD: OpenCV histogram (if face_recognition failed)
    if 'face_recognition' not in verification_result:
        print("\n📊 Using OpenCV histogram fallback...")
        result, error = compare_faces_opencv(cnic_picture, live_face)
        if result:
            verification_result['opencv_histogram'] = result
            verification_result['methods_tried'].append('opencv_histogram')
            verification_result['final_verification'] = result['is_match']
            verification_result['confidence'] = result['similarity']
    
    # If no method worked
    if 'final_verification' not in verification_result:
        verification_result['final_verification'] = False
        verification_result['error'] = 'Could not perform face comparison'
        verification_result['confidence'] = 0
    
    return verification_result

def capture_live_face():
    """Capture face from webcam"""
    print("\n📸 Starting webcam for live face capture...")
    print("   Press 'C' to capture your face")
    print("   Press 'Q' to quit")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return None
    
    cv2.namedWindow('Live Face Capture - Press C to Capture', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Live Face Capture - Press C to Capture', 800, 600)
    
    captured_face = None
    face_coords = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture frame")
            break
        
        display_frame = frame.copy()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display_frame, "Face Detected! Press 'C' to capture", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            face_coords = (x, y, w, h)
        
        cv2.putText(display_frame, "Press 'C' to Capture Face", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press 'Q' to Quit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Live Face Capture - Press C to Capture', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c') or key == ord('C'):
            if face_coords is not None:
                x, y, w, h = face_coords
                captured_face = frame[y:y+h, x:x+w]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                face_filename = f"captured_face_{timestamp}.jpg"
                cv2.imwrite(face_filename, captured_face)
                print(f"✅ Face captured and saved as '{face_filename}'")
                cv2.putText(display_frame, "FACE CAPTURED!", (x, y+h+30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Live Face Capture - Press C to Capture', display_frame)
                cv2.waitKey(1000)
                break
            else:
                print("⚠️ No face detected! Please position your face properly.")
        
        elif key == ord('q') or key == ord('Q'):
            print("⏹️ Live capture cancelled")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return captured_face

def process_cnic_front(front_imagee, cnic_processor):
    """Process all CNIC fields from front side using YOLO detection"""
    print("🚀 Starting CNIC Front Side Processing...")
    
    detections = detect_cnic_fields(front_imagee, cnic_processor.model, cnic_processor.class_names)
    
    if len(detections) == 0:
        print("❌ No CNIC fields detected!")
        return {}, [], None
    
    print(f"✅ Detected {len(detections)} CNIC fields")
    
    cnic_picture, picture_bbox = extract_picture_from_cnic(front_imagee, detections)
    if cnic_picture is not None:
        cv2.imwrite('cnic_picture_extracted.jpg', cnic_picture)
        print(f"📸 CNIC picture extracted and saved to 'cnic_picture_extracted.jpg'")
    
    extracted_data = {}
    all_filtered_data = []
    
    detections.sort(key=lambda x: x['bbox'][1])
    
    for detection in detections:
        field_name = detection['class_name']
        if field_name == 'picture':
            continue
            
        text = extract_text_from_roi(front_imagee, detection['bbox'], cnic_processor.reader, field_name)
        
        if text:
            display_names = {
                'name': 'Name',
                'fname': 'Father Name', 
                'id': 'ID Card Number',
                'bdate': 'Date of Birth',
                'idate': 'Date of Issue',
                'edate': 'Date of Expiry',
                'gender': 'Gender',
                'country': 'Country',
                'picture': 'Picture',
                'CNIC-HHMI': 'CNIC Header'
            }
            
            display_name = display_names.get(field_name, field_name)
            if field_name == 'CNIC-HHMI':
                print(f"⏭️  Skipping {display_name} (contains multiple fields)")
                continue
            extracted_data[display_name] = text
            all_filtered_data.append(text)
            print(f"📋 {display_name}: {text} (conf: {detection['confidence']:.2f})")
        else:
            print(f"⚠️ {field_name}: No text extracted (conf: {detection['confidence']:.2f})")
    
    return extracted_data, all_filtered_data, cnic_picture

def display_detected_fields(image, detections, window_name='Detected CNIC Fields'):
    """Display image with detected fields"""
    display_image = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection['bbox'])
        class_name = detection['class_name']
        confidence = detection['confidence']
        
        cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(display_image, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow(window_name, display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def validate_cnic_data(data):
    """Validate and format extracted CNIC data"""
    validated = {}
    
    if 'ID Card Number' in data:
        cnic = data['ID Card Number']
        cnic_clean = re.sub(r'[^\d-]', '', cnic)
        if re.match(r'\d{5}-\d{7}-\d{1}', cnic_clean):
            validated['ID Card Number'] = cnic_clean
        else:
            validated['ID Card Number'] = cnic
    
    date_fields = {'Date of Birth': 'bdate', 'Date of Issue': 'idate', 'Date of Expiry': 'edate'}
    
    for field_name, _ in date_fields.items():
        if field_name in data:
            date_str = data[field_name]
            date_patterns = [
                (r'(\d{2})\.(\d{2})\.(\d{4})', r'\1.\2.\3'),
                (r'(\d{2})-(\d{2})-(\d{4})', r'\1.\2.\3'),
                (r'(\d{2})/(\d{2})/(\d{4})', r'\1.\2.\3'),
            ]
            
            for pattern, replacement in date_patterns:
                match = re.search(pattern, date_str)
                if match:
                    validated[field_name] = match.group(0).replace('-', '.').replace('/', '.')
                    break
            else:
                validated[field_name] = date_str
    
    for key, value in data.items():
        if key not in validated:
            validated[key] = value
    
    return validated

def save_results(data, filename='cnic_front_data.csv'):
    """Save extracted data to CSV, JSON, and text files"""
    if not data:
        print("No data to save")
        return
    
    validated_data = validate_cnic_data(data)
    
    csv_data = []
    for field_name, text in validated_data.items():
        csv_data.append({'Field': field_name, 'Value': text})
    
    df = pd.DataFrame(csv_data)
    
    try:
        df.to_csv(filename, index=False)
        print(f"💾 Data saved to {filename}")
    except PermissionError:
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        new_filename = filename.replace('.csv', f'_{timestamp}.csv')
        df.to_csv(new_filename, index=False)
        print(f"⚠️  Original file locked. Data saved to {new_filename}")
        filename = new_filename
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")
        return
    
    try:
        json_filename = filename.replace('.csv', '.json')
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(validated_data, f, indent=2, ensure_ascii=False)
        print(f"💾 JSON data saved to {json_filename}")
    except Exception as e:
        print(f"⚠️  Could not save JSON: {e}")
    
    try:
        txt_filename = filename.replace('.csv', '.txt')
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("  CNIC FRONT SIDE EXTRACTED DATA\n")
            f.write("=" * 50 + "\n\n")
            
            field_order = ['Name', 'Father Name', 'ID Card Number', 'Date of Birth', 
                          'Date of Issue', 'Date of Expiry', 'Gender', 'Country']
            
            for field in field_order:
                if field in validated_data:
                    f.write(f"{field:20s}: {validated_data[field]}\n")
            
            for field_name, text in validated_data.items():
                if field_name not in field_order:
                    f.write(f"{field_name:20s}: {text}\n")
            
            f.write("\n" + "=" * 50 + "\n")
        print(f"💾 Formatted text summary saved to '{txt_filename}'")
    except Exception as e:
        print(f"⚠️  Could not save text file: {e}")

def create_annotated_image(image, detections, output_path='cnic_annotated.jpg'):
    """Create and save annotated image"""
    annotated_image = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection['bbox'])
        class_name = detection['class_name']
        confidence = detection['confidence']
        
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(annotated_image, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(annotated_image, label, (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    cv2.imwrite(output_path, annotated_image)
    print(f"🖼️ Annotated image saved as '{output_path}'")

def capture_cnic_via_webcam():
    """Capture CNIC card using webcam"""
    print("\n📷 Webcam CNIC Capture Mode")
    print("="*60)
    print("Instructions:")
    print("1. Hold your CNIC card in front of the camera")
    print("2. Make sure the entire card is visible and well-lit")
    print("3. Press 'C' to capture the image")
    print("4. Press 'Q' to quit")
    print("="*60)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return None
    
    cv2.namedWindow('Capture CNIC Card - Press C to Capture', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Capture CNIC Card - Press C to Capture', 800, 600)
    
    captured_image = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture frame")
            break
        
        display_frame = frame.copy()
        h, w = frame.shape[:2]
        border_size = 30
        cv2.rectangle(display_frame, (border_size, border_size), 
                     (w - border_size, h - border_size), (0, 255, 0), 2)
        
        cv2.putText(display_frame, "Position CNIC card inside the green box", 
                   (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press 'C' to Capture", 
                   (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press 'Q' to Quit", 
                   (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Capture CNIC Card - Press C to Capture', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c') or key == ord('C'):
            captured_image = frame.copy()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_cnic_{timestamp}.jpg"
            cv2.imwrite(filename, captured_image)
            print(f"✅ CNIC image captured and saved as '{filename}'")
            cv2.putText(display_frame, "IMAGE CAPTURED!", 
                       (w//2 - 100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.imshow('Capture CNIC Card - Press C to Capture', display_frame)
            cv2.waitKey(1000)
            break
        
        elif key == ord('q') or key == ord('Q'):
            print("⏹️ Capture cancelled")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return captured_image

def main_menu():
    """Display main menu for user interaction"""
    print("\n" + "="*60)
    print("        CNIC PROCESSING SYSTEM WITH LIVE VERIFICATION")
    print("="*60)
    print("\nSelect Mode:")
    print("1. Process CNIC from file + Live face verification")
    print("2. Capture CNIC via webcam + Live face verification")
    print("3. Process CNIC from file only (no verification)")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        if choice in ['1', '2', '3', '4']:
            return choice
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == '__main__':
    print("🔄 Initializing CNIC Processor...")
    try:
        cnic_processor = CNICProcessor('runs/detect/train3/weights/best.pt')
        print("✅ Custom YOLO model loaded successfully!")
    except Exception as e:
        print(f"❌ Custom model not found: {e}")
        print("⚠️ Using default YOLO model (will detect generic objects)")
        cnic_processor = CNICProcessor('yolov8n.pt')
        cnic_processor.class_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light'
        }
    
    while True:
        choice = main_menu()
        
        if choice == '1':
            print("\n" + "="*60)
            print("MODE 1: Process CNIC from file + Live face verification")
            print("="*60)
            
            cnic_path = input("\nEnter path to CNIC image file (or press Enter for default): ").strip()
            if not cnic_path:
                cnic_path = './Dataset/obd.jpeg'
            
            if not os.path.exists(cnic_path):
                print(f"❌ File not found: {cnic_path}")
                continue
            
            front_image = cv2.imread(cnic_path)
            if front_image is None:
                print(f"❌ Could not read image: {cnic_path}")
                continue
            
            front_detections = detect_cnic_fields(front_image, cnic_processor.model, cnic_processor.class_names)
            if len(front_detections) == 0:
                print("❌ No CNIC fields detected in the image")
                continue
            
            print(f"✅ Detected {len(front_detections)} CNIC fields")
            create_annotated_image(front_image, front_detections, 'cnic_detected_fields.jpg')
            extracted_data, all_data, cnic_picture = process_cnic_front(front_image, cnic_processor)
            
            if not extracted_data:
                print("❌ No data extracted from CNIC")
                continue
            
            print("\n" + "="*60)
            print("LIVE FACE VERIFICATION")
            print("="*60)
            
            live_face = capture_live_face()
            
            if live_face is not None:
                verification_result = verify_face_live(cnic_picture, live_face)
                
                if verification_result and 'final_verification' in verification_result:
                    extracted_data['Face Verification'] = '✅ MATCH' if verification_result['final_verification'] else '❌ NO MATCH'
                    if 'confidence' in verification_result and verification_result['confidence'] != 'N/A':
                        extracted_data['Face Similarity'] = f"{verification_result['confidence']:.2f}%"
                    
                    print("\n" + "="*60)
                    print("FACE VERIFICATION SUMMARY")
                    print("="*60)
                    print(f"   Status: {'✅ VERIFIED' if verification_result['final_verification'] else '❌ NOT VERIFIED'}")
                    if 'confidence' in verification_result and verification_result['confidence'] != 'N/A':
                        print(f"   Similarity: {verification_result['confidence']:.2f}%")
                    print(f"   Methods used: {', '.join(verification_result.get('methods_tried', []))}")
                    print("="*60)
            
            save_results(extracted_data)
            
        elif choice == '2':
            print("\n" + "="*60)
            print("MODE 2: Capture CNIC via webcam + Live face verification")
            print("="*60)
            
            captured_cnic = capture_cnic_via_webcam()
            if captured_cnic is None:
                print("❌ Failed to capture CNIC image")
                continue
            
            front_detections = detect_cnic_fields(captured_cnic, cnic_processor.model, cnic_processor.class_names)
            if len(front_detections) == 0:
                print("❌ No CNIC fields detected in the captured image")
                print("   Please try again with better lighting and positioning")
                continue
            
            print(f"✅ Detected {len(front_detections)} CNIC fields")
            create_annotated_image(captured_cnic, front_detections, 'captured_cnic_annotated.jpg')
            extracted_data, all_data, cnic_picture = process_cnic_front(captured_cnic, cnic_processor)
            
            if not extracted_data:
                print("❌ No data extracted from CNIC")
                continue
            
            print("\n" + "="*60)
            print("LIVE FACE VERIFICATION")
            print("="*60)
            
            live_face = capture_live_face()
            
            if live_face is not None:
                verification_result = verify_face_live(cnic_picture, live_face)
                
                if verification_result and 'final_verification' in verification_result:
                    extracted_data['Face Verification'] = '✅ MATCH' if verification_result['final_verification'] else '❌ NO MATCH'
                    if 'confidence' in verification_result and verification_result['confidence'] != 'N/A':
                        extracted_data['Face Similarity'] = f"{verification_result['confidence']:.2f}%"
                    
                    print("\n" + "="*60)
                    print("FACE VERIFICATION SUMMARY")
                    print("="*60)
                    print(f"   Status: {'✅ VERIFIED' if verification_result['final_verification'] else '❌ NOT VERIFIED'}")
                    if 'confidence' in verification_result and verification_result['confidence'] != 'N/A':
                        print(f"   Similarity: {verification_result['confidence']:.2f}%")
                    print(f"   Methods used: {', '.join(verification_result.get('methods_tried', []))}")
                    print("="*60)
            
            save_results(extracted_data)
            
        elif choice == '3':
            print("\n" + "="*60)
            print("MODE 3: Process CNIC from file only")
            print("="*60)
            
            cnic_path = input("\nEnter path to CNIC image file (or press Enter for default): ").strip()
            if not cnic_path:
                cnic_path = './Dataset/obd.jpeg'
            
            if not os.path.exists(cnic_path):
                print(f"❌ File not found: {cnic_path}")
                continue
            
            front_image = cv2.imread(cnic_path)
            if front_image is None:
                print(f"❌ Could not read image: {cnic_path}")
                continue
            
            front_detections = detect_cnic_fields(front_image, cnic_processor.model, cnic_processor.class_names)
            if len(front_detections) == 0:
                print("❌ No CNIC fields detected in the image")
                continue
            
            print(f"✅ Detected {len(front_detections)} CNIC fields")
            display_detected_fields(front_image, front_detections, 'Detected CNIC Fields')
            create_annotated_image(front_image, front_detections, 'cnic_detected_fields.jpg')
            extracted_data, all_data, cnic_picture = process_cnic_front(front_image, cnic_processor)
            
            if extracted_data:
                save_results(extracted_data)
            else:
                print("❌ No data extracted from CNIC")
        
        elif choice == '4':
            print("\n👋 Exiting CNIC Processing System. Goodbye!")
            break
        
        continue_choice = input("\nDo you want to process another CNIC? (y/n): ").strip().lower()
        if continue_choice != 'y':
            print("\n👋 Exiting CNIC Processing System. Goodbye!")
            break
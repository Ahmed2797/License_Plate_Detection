import re
import json
import os
from datetime import datetime
import sqlite3
import numpy as np
from paddleocr import PaddleOCR



# ----------------------------
# PaddleOCR init for license plates
# ----------------------------
ocr = PaddleOCR()

def paddle_ocr_plate(frame, x1, y1, x2, y2, confidence_threshold=0.1):
    roi = frame[y1:y2, x1:x2]  # crop license plate region
    results = ocr.predict(roi)     # run OCR
    extracted_texts = []

    # Debug print
    #print('results***********',results)
    


    # Handle dict output
    first_result = results[0]
    print('results***********',first_result)

    if isinstance(first_result, dict):
        rec_texts = first_result.get("rec_texts", [])
        rec_scores = first_result.get("rec_scores", [])
        print("Detected text:*****", rec_texts)
        print("Confidence:******", rec_scores)

        for text, score in zip(rec_texts, rec_scores):
            if np.isnan(score):  # handle NaN
                score = 0

            if score >= confidence_threshold:
                extracted_texts.append(text)

    # Combine all detected text and clean
    final_text = "".join(extracted_texts)
    final_text = re.sub(r'\W+', '', final_text)  
    final_text = final_text.replace("O", "0")    

    return final_text




# Save JSON file 
def save_json(license_plate, starttime, endtime):
    interval_data = {
        'starttime': starttime.isoformat(),
        'endtime': endtime.isoformat(),
        'license_plate': list(license_plate)
    }

    os.makedirs('project/json',exist_ok=True)

    # Save interval JSON uniquely
    interval_data_path = f'project/json/output_{datetime.now().isoformat()}.json'
    with open(interval_data_path, 'w') as f:
        json.dump(interval_data, f, indent=2)

    # Append to JSON file
    complete_data_path = 'project/json/license_plate_data.json'
    if os.path.exists(complete_data_path):
        with open(complete_data_path, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_data.append(interval_data)

    with open(complete_data_path, 'w') as f:
        json.dump(existing_data, f, indent=2)

    save_json_dbbase(license_plate,starttime,endtime)




# Save into SQLite DB
def save_json_dbbase(license_plate, starttime, endtime):
    conn = sqlite3.connect('license_plate.db')
    cursor = conn.cursor()

    # Create table if not exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            starttime TEXT,
            endtime TEXT,
            license_plate TEXT
        )
    ''')

    for plate in license_plate:
        cursor.execute('''
            INSERT INTO history (starttime, endtime, license_plate)
            VALUES (?, ?, ?)
        ''', (starttime.isoformat(), endtime.isoformat(), plate))

    conn.commit()
    conn.close()

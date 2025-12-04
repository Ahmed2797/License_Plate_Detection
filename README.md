# License_Plate_Detection

reate a folder in system

licences_plate_detect

## create environment

    conda create -n lc_plate python=3.12
    conda activate lc_plate
    conda deactivate 
    pip install -r requirements.txt

## create folder in code-editor

    bash setup.sh

    python sql.db.py

    uvicorn app:app --reload

## notics

My model was trained with a small number of epochs. If possible, please increase the number of epochs to achieve better detection.

## Explaination

🚗 Automating Vehicle License Plate Detection with Python! 🟢

I’m excited to share my latest project: a Full License Plate Detector built using YOLO and EasyOCR — fully Colab-ready! 🎉

💡 What it does:

Detects vehicles in videos frame by frame.

Reads license plates with EasyOCR (lightweight and efficient).

Saves all detected plates into JSON files and a SQLite database automatically.

Annotates videos with detected plates and confidence scores.

Customizable: detection confidence, save interval, and display options.

Why I built it:
This tool can be a great starting point for smart parking systems, traffic monitoring, or vehicle tracking solutions. It demonstrates how AI can automate tasks that used to require hours of manual work.

🔗 Tech stack: Python | YOLO | EasyOCR | OpenCV | SQLite

Excited to see how this can be improved further and applied in real-world scenarios! 🚀

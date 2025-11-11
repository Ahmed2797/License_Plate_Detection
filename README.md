# License_Plate_Detection

# create a folder in system
licences_plate_detect 

# create environment
conda create -n lc_plate python=3.12
conda activate lc_plate
conda deactivate 

# Install
pip install -r requirements.txt 

# create folder in code-editor 
bash setup.sh 

python sql.db.py 


uvicorn app:app --reload 

# notics
My model was trained with a small number of epochs. If possible, please increase the number of epochs to achieve better detection.



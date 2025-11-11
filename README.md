# License_Plate_Detection

<h2>create a folder in system</h2>
licences_plate_detect 

<h2>create environment</h2>

conda create -n lc_plate python=3.12
conda activate lc_plate
conda deactivate 

<h2>Install</h2>
pip install -r requirements.txt 

<h2>create folder in code-editor</h2> 
bash setup.sh 

python sql.db.py 


uvicorn app:app --reload 

<h2>notics</h2>
My model was trained with a small number of epochs. If possible, please increase the number of epochs to achieve better detection.



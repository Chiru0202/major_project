How to run:(Run these commands in your terminal)
py -m venv .venv #(creates a virtual environment)
.\.venv\Scripts\Activate.ps1                      #(activate the created environment ) (venv) should appear to the right side
python -m pip install --upgrade pip               #(installs and upgrades if any upgrades are needed)
pip install -r requirements.txt                   #(Installs all the requirements needed to run this project)
python app.py                                     #(check your local host http://127.0.0.1:5000)

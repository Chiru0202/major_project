## How to Run

Run these commands in your terminal:

```powershell
# Creates a virtual environment
py -m venv .venv

# Activates the created environment
.\.venv\Scripts\Activate.ps1

# (venv) should appear to the right side of your terminal

# Installs/upgrades pip if needed
python -m pip install --upgrade pip

# Installs all the requirements needed to run this project
pip install -r requirements.txt

# Runs the application
python app.py
```

The application will be available at:

**http://127.0.0.1:5000**

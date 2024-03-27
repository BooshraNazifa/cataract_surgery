# Prerequisites:

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

Install them using the following commands:

```
sudo apt-get install python3.8
sudo apt-get install python3-pip
python3 -m pip install --user virtualenv
```

# Clone the repository:

```
git clone https://github.com/BooshraNazifa/cataract_surgery.git
cd cataract_surgery
```

# Set up a virtual environment (optional but recommended):

```
python3 -m venv venv
source venv/bin/activate
```

# Install the required packages:

```
pip install -r requirements.txt
```


# Running the Application:
## Change the file paths

```
python frame_extraction.py
python phase_recognition.py
```
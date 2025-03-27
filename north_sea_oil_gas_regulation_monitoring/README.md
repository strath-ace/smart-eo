# FRIL

Satellite data work for FRIL Oil and Gas Regulation Monitoring project


### Setup (For linux)

1. Create virtual environment for python

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Organize data into FIRMS, NSTA and SENTINEL folders

3. Run `raw_to_np.py` in SENTINEL folder

4. If youd like to collect more SENTINEL data using the sentinelhub python api, you will need to copy and fill in the `.env_template` file and rename is `.env`.


### Run processing code

Run `python src/<codetorun>.py`.

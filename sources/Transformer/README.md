cd sources
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\Transformer\requirements.txt

streamlit run .\Transformer\app.py

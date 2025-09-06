# Data Agent Project

This repo contains an experimental AI agent.  
Development flow:
- Experiment in `/notebooks`
- Move stable code into `/data_agent`
- Add tests under `/tests` (sibling of data_agent)
- Track everything with GitHub
- Execute `main.py` for local development

## Running in Colab
Clone repo inside Colab and install, with the following 
   ```python
      !git clone https://{github_token}@github.com/DonErnesto/data-agent.git /content/data-agent

      %cd /content/data-agent
      !mkdir tmp #to write pickled litellm completions.
      !pip install data-agent
      !pip install -r requirements-colab.txt

## Running locally
pip install -r requirements.txt
!cd data-agent
pip install -e . 
pytest #to run tests
python main.py


# Data Agent Project

This repo contains an experimental AI agent.  
Development flow:
- Experiment in `/notebooks`
- Move stable code into `/data_agent`
- Add tests under `/tests` (sibling of data_agent)
- Track everything with GitHub
- Execute `main.py` for local development (new: using `make run`, see below)

## Running in Colab
Clone repo inside Colab and install, with the following 
```python
   !git clone https://{github_token}@github.com/DonErnesto/data-agent.git /content/data-agent

   %cd /content/data-agent
   !mkdir tmp
   !pip install data-agent
   !pip install -r requirements-colab.txt
 ```

## Development Workflow

This project comes with a `Makefile` to simplify common tasks.

### Installation

To set up the project with all dependencies:

```bash
make install
```
This will:
- Install dependencies from requirements.txt
- Install the data_agent package in editable (-e) mode

## Running the Project
To run the main entry point:

```bash
make run
```

## Testing
To run the test suite (with cache cleared automatically):

```bash
make test
```

## Cleaning
To remove Python caches and pytest caches:

```bash
make clean
```

## Notes
Always use make install after updating dependencies in requirements.txt.
With editable install (pip install -e .), changes to the source code are immediately reflected without reinstalling.
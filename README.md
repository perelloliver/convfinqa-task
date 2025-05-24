# TODO Prior to Submission:
- Proofread report
- Run evals on chosen models and write in results / discuss
- Clean up comments
- Test containerized run, check Dockerfile, update requirements

# ConvFinQA Task

This repository implements a financial conversation QA agent using the ConvFinQA dataset.

## Main components:
- Agent: A simple agent which uses the OpenAI agents SDK.
- Evals: Evaluation metrics for agent performance compared to dataset ground truth.
- Models: Data models for parsing dataset and storing agent responses.
- Main: Main script to run the agent and evaluation.
- Parsing: Data parsing script to convert ConvFinQA dataset into an easily usable format for prompting and evaluation.
- Report: A markdown file containing a written report of the task: methodology, outcomes, assessment and next actions.

## Getting Started

### Requirements
- Docker (recommended)
- Or: Python 3.10+, pip, virtualenv

### Quickstart (Docker)

1. Build the image:
   ```sh
   docker build -t convfinqa .
   ```
2. Run the container (choose mode: `tiny` for quick test, `test` for full eval):
   ```sh
   docker run --rm -v $(pwd):/app convfinqa --mode tiny
   # or
   docker run --rm -v $(pwd):/app convfinqa --mode test
   ```

### Quickstart (Local)

1. Install dependencies:
   ```sh
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run the main script:
   ```sh
   python -m app.main --mode tiny   # For a quick test
   python -m app.main --mode test   # For full evaluation
   ```

## Arguments
- `--mode`: Either `tiny` (quick test) or `test` (full evaluation). Default: `tiny`.

## Output
- Evaluation metrics are printed to the terminal.
- Agent responses are saved to `data/responses.csv`.

## Project Structure
- `app/` - Main application code with agent, evals, models, and main.
- `data/` - Datasets, outputs, data parsing script.

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
- An OpenAI API Key
- Or: Python 3.10+, pip, virtualenv

### Quickstart (Docker)

1. Set your OpenAI API key in the .env file (use .env.example as a template) or as an environment variable in your terminal.

2. Build the image:
   ```sh
   docker build -t convfinqa .
   ```

3. Run the container (choose mode: `tiny` for quick test, `test` for full eval):
   ```sh
   docker run --rm -v $(pwd):/app convfinqa --mode tiny
   # or
   docker run --rm -v $(pwd):/app convfinqa --mode test
   ```

### Quickstart (Local)

1. Set your OpenAI API key in the .env file (use .env.example as a template) or as an environment variable in your terminal.

2. Install dependencies:
   ```sh
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```sh
   python -m app.main --mode tiny   # For a quick test
   python -m app.main --mode test   # For full evaluation
   ```

## Arguments
- `--mode`: Either `tiny` (quick test), `test` (run test set), or `full` (run full dataset). Default: `tiny`.

## Output
- Evaluation metrics are printed to the terminal.
- Agent responses are saved to `data/responses.csv`.
- Parsed data for future use is saved to data/formatted_dataset in train/test/val splits and the full dataset.

## Project Structure
- `app/` - Main application code with agent, evals, models, and main.
- `data/` - Datasets - original, parsed, and responses.

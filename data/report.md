
# Writeup

## Task overview:

Build an LLM agent driven prototype that can answer conversational questions about financial documents, using the ConvFinQA dataset.

## Requirements

- Plain RAG unsuitable.
- Must use ConvFinQA/train.json
- Must work with semi-structured text e.g dataset queries are partially markdown.
- Agent should return answer/program responses in the same format as original dataset for evaluation purposes.

## Deliverables

- Working LLM agent for answering conversational financial questions with a numeric answer and a natural-language program.
- A script to run and reproduce evaluations.
- Clean, clear repository with documentation, containerization and placeholder CI/CD.
- A report with the writeup of method, decisions, design, evaluation results and a system assesment.

## Core actions

- [x] Wrangle ConvFinQA dataset into a usable format reproducing multi-turn conversations via an LLM API.
- [x] Develop a basic agent capable of answering multi-turn conversations from the ConvFinQA dataset.
- [x] Wrap agent in a script to run and reproduce evaluations, starting from the raw dataset.
- [x] Containerize, CI/CD, documentation, requirements, etc.
- [ ] Evaluate at least two different models on the test set.
- [ ] Add a user entrypoint for interactive use.

## Method

#### High level design


#### Solution

First, I laid out a basic system design and defined a core data schema from which to model our dataset. The model, ```DatasetConvQaParsed```, is a simple representation of a turn in a conversation, with separate, parsed fields for us to use in prompt chaining and evaluations.

```python
class DatasetConvQaParsed(BaseModel):
    """Data model for parsing dataset"""
    context: str = Field(description="pre_text, table, post_text structured in MD")
    qa_history: Optional[List[Dict[str, str]]] = Field(description="QA history of user<> question, system<> answer pairs", default=[])
    current_question: str = Field(description="Current question")
    gold_program: str = Field(description="program from dataset")
    gold_answer: Any = Field(description="correct answer")
    id: str = Field(description="conversation id")
    turn_index: int = Field(description="turn count")
    type: str = Field(description="Type I or II")
```

In ```parsing.py```, I wrote functions to parse the raw dataset into this schema, split it into train/test/validation sets, save our formatted split data to CSV and return a dataframe for live use.

Then, I iteratively implemented a basic agent. I usually work with LangGraph, but considering the time-scope of this project, I chose to use the OpenAI Agents SDK.

This agent is equipped with a set of tools, largely similar to those used in the original ConvFinQA paper. These tools output a ```ToolReturn``` object, which matches the format of the dataset to provide both a numeric answer and a program. 

This ```ToolReturn``` object is then used to generate the final output for the agent in the ```Answer``` model.

I added one additional tool, direct_retrieval. This tool is used by the LLM to retrieve a figure directly from the context. I added this tool to ensure we uniformly process and output results, providing a direct logical path for the agent even for non-mathematic queries.


#### Model choice
- Practical factors:
    - Latency
    - Cost-effectiveness
    - Compatible with OpenAI Agents SDK
    - Supports tool use and structured outputs
    - Good logistical reasoning capability: performs well in mathematical or coding environments.

- Exploratory factors:
    - Higher reasoning capabilities
    - Supports RLF fine-tuning

Initially, I selected o3-mini for its overall good performance and cost-effectiveness compared to other reasoning models. However, I quickly found the prototyping cost and latency to be unsuitable for this task.

While I believe that reasoning models are a better fit for this task, I am also aware that raw model performance is not an essential part of this project. 

As such, I swapped from o3-mini to gpt-4.1-mini, which is a more cost-effective and high latency model, to better suit the prototyping process. Since the tools involved in question-answering are relatively simple and the path is defined, I expect the model to perform well enough for this task.

If I had more time and resources to experiment, I'd definitely test and evaluate multiple reasoning models and explore fine-tuning. The original paper notes that a neural symbolic approach was more successful in their original experiments; although it has been some years since the paper release, and model improvements have been significant, I'm curious to see what kind of performance we could achieve with a fine-tuning approach, notably reinforcement learning fine-tuning on a reasoning model like o4-mini.


#### Evaluation metric choices and reasoning

I selected the following evaluation metrics:
    i. Execution accuracy: was the answer right?
    iii. Turn degradation: what is the degradation slope of accuracy over turns?

Execution accuracy is a key metric - getting the answer right is arguably the most important metric. Since our 'program' is programmatically generated via tool use, it's always equivalent if the answer is correct, so we don't require metrics for the program value.

Turn degradation is a metric I added to evaluate the agent's performance over time. It's not a key metric, but it's still important to evaluate - it's a good indicator of whether the agent is deteriorating in performance over time. Turn degradation is a real problem for domain use: users expect consistent performance over time, and a drop in performance can lead to mistakes and trust degradation for our users.

If I had more time, I also would have added domain-specific error analysis to see what types of errors were most common - revenue/profit confusions, etc.

#### Evaluation results
Discuss results here

#### Weaknesses and limitations

I ran out of time before I could finish everything I wanted to do, so the system has some limitations. There is no true user entrypoint - our main script is only for running and reproducing evaluations, not for interactive use. If I had more time to iterate on this, I would have added a terminal-based entrypoint for interactive use with a random dataset sample.

The overall performance is not particularly good. {{Discuss metrics from full eval run}}. Turn-based degradation is significant, with an average of {x} drop in accuracy on a per-turn basis. Likewise, overall accuracy sits at {y}.

With such a simple system, performance is largely reliant on the model we use. It's not a fully comprehensive system in that respect - just a workflow for handling multi-turn conversations based on the ConvFinQA dataset. If I had more time, I'd evaluate several different models and explore fine-tuning to see if I could improve the output quality.


#### Strengths

The agent is simple and easy to use, with overall good latency and cost-effectiveness.

The data structure is comprehensive and clear.

#### What I'd do differently
This is what I would do differently if I had more time, or if I carry on with this project in the future, in order of importance:

- Implement CI/CD, just a basic GitHub workflow.
- Improve error handling and logging to prevent data loss and overall improve system performance.
- Evaluate different reasoning models to identify the best choice for this system.
- Add functions to measure latency and tokens per turn.
- Create an entrypoint for user queries.
- Evaluate results for domain-specific error analysis to identify common points of failure.
- Add a more comprehensive, domain-specific toolset, possibly even connect to a relevant MCP server, to provide more context and user capabilities beyond that of the dataset.
- Test out RLF fine-tuning when exploring other models.

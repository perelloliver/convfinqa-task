
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
- [x] Evaluate at least two different models on the test set.

## Method

**Key Design Decisions**

- We will develop an agent, capable of parsing structured and semi-structured contextual information, understanding user queries, calling tools and outputting responses in the required format.
- The dataset needs to be easily parsable and uniform in order to feed it to the agent.
- Agent tools should be constrained to output answers in the required format, ensuring response uniformity.
- We will process conversations, and within conversations we will process turns.
- We will use chat history persistence between turns.
- We will use semi-structured text, as the dataset queries are partially markdown.
- We will use a reasoning model, as this is a reasoning task and we lack the scope for fine-tuning.
- For comparison, we will also use a non-reasoning model.
- We should have options for what kind of evalutions we want to perform: a small test, the full test set, the entire dataset, etc.
- We should be able to change the model we are using with some ease.

**Implementation**

First, I laid out a basic system design to follow, and defined a core data schema from which to model our dataset. 

The model, ```DatasetConvQaParsed```, is a simple representation of a turn in a conversation, with separate, parsed fields for us to use in prompt chaining and evaluations.

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

This agent is equipped with a set of tools, largely similar to those used in the original ConvFinQA paper with a couple of additions - ```percentage``` and ```direct_retrieval```. The ```direct_retrieval``` tool is used by the LLM to retrieve a figure directly from the context. I added this tool to ensure we uniformly process and output results, providing a direct logical path for the agent even for non-mathematic queries.

These tools output a ```ToolReturn``` object, which matches the format of the dataset to provide both a numeric answer and a program. 
This ```ToolReturn``` object is then used to generate the final output for the agent in the ```Answer``` model.


#### Choosing models
- Practical factors:
    - Latency
    - Cost-effectiveness
    - Compatible with OpenAI Responses API, for use in agents SDK
    - Good logistical reasoning capability: performs well in mathematical or coding environments.

- Exploratory factors:
    - Higher reasoning capabilities
    - Supports RLF fine-tuning

I would have liked to use o3/o4 and their respective mini models for their advanced reasoning capabilities, but unfortunately working from my personal account I don't have access to these models. Instead, I selected o1-pro, which is specifically designed for use in the Responses API for agentic solutions, and gpt-4.1 to compare reasoning versus non-reasoning model capabilities.

I primarily developed with gpt-4.1-mini, which is more cost-effective and has better latency for prototyping. Once I had a working prototype, I evaluated the system with o1-pro and gpt-4.1.

If I had more time and resources to experiment, I'd definitely test and evaluate multiple reasoning models and explore fine-tuning.

The original paper notes that a neural symbolic approach was more successful in their original experiments; although it has been some years since the paper release, and model improvements have been significant, I'm curious to see what kind of performance we could achieve with a fine-tuning approach, notably reinforcement learning fine-tuning on a reasoning model like o4-mini.


#### Evaluation metric choices and reasoning

I selected the following evaluation metrics:
    i. Execution accuracy: was the answer right?
    iii. Turn degradation: what is the average accuracy for each turn, and what is the degradation slope of accuracy over turns?

Execution accuracy is a key metric - getting the answer right is arguably the most important metric. Since our 'program' is programmatically generated via tool use, it's always equivalent if the answer is correct, so we don't require metrics for the program value.

Turn degradation is a metric I added to evaluate the agent's performance over time. It's not a key metric, but it's still important to evaluate - it's a good indicator of whether the agent is deteriorating in performance over time. Turn degradation is a real problem for domain use: users expect consistent performance over time, and a drop in performance can lead to mistakes and trust degradation for our users.

If I had more time, I also would have added domain-specific error analysis to see what types of errors were most common - revenue/profit confusions, etc.

#### Results (discussed below)



#### Weaknesses and limitations

I ran out of time before I could finish everything I wanted to do, so the system definitely has some limitations. 

There is no true interactive entrypoint - our main script is only for running and reproducing evaluations, not for interactive use. If I had more time to iterate on this, I would have added a terminal-based entrypoint for interactive use with a random dataset sample, so the user can see the data, ask questions, and receive answers from the agent in a real multi-turn conversation.

Due to cost and time constraints, I was only able to run evaluations on ```tiny``` mode, which runs over 50 examples. I ran this tiny evaluation for all models I selected. From this small sample, the overall performance is not particularly good. {{Discuss metrics from full eval run}}. Turn-based degradation is significant, with an average of {x} drop in accuracy on a per-turn basis. Likewise, overall accuracy sits at {y}.

With such a simple system, performance is largely reliant on the model we use. It's not a fully comprehensive system in that respect - just a workflow for handling multi-turn conversations based on the ConvFinQA dataset. If I had more time, not only would I evaluate multiple models for best performance, but I would add complexity and robustness to the agent for improved output.

For example, I would constrain tool calling to ensure the agent does not make redundant or overcomplicated tool calls, and I would run conversation persistence between turns using the chat ID, for elegance and to maintain a clearer context window. To improve multi-turn performance, I would explore post-turn summarization in place of persistent chat history acting as memory. 

#### Strengths

The agent is simple and easy to use, with overall good latency and cost-effectiveness. The code is modular and ready to iterate on for a more robust system. It's simple to run experiments of varying kinds with the ```mode``` parameter, which allows for easy switching between train/test/validation sets and a ```tiny``` option for a quick run. Likewise, it's simple to switch models by updating the ```llm``` parameter for our agent.

The data structure is comprehensive and clear, which is hugely beneficial for building a more robust system or exploring fine-tuning approaches in future. Having control over micro-elements of the data to construct well-formed prompts is a strength of this system.

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

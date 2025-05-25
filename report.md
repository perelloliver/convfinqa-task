
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
- [x] Documentation, requirements, quick and easy entrypoint to run.
- [x] Evaluate at least two different models on the test set.

## Method

**Primary Design Decisions (High Level)**

- We will develop an agent, capable of parsing structured and semi-structured contextual information, understanding user queries, calling tools and outputting responses in the required format.
- The dataset needs to be easily parsable and uniform in order to feed it to the agent.
- Agent tools should be constrained to output answers in the required format, ensuring response uniformity.
- We will process conversations, and within conversations we will process turns.
- We will use chat history persistence between turns.
- We will use semi-structured text, as the dataset queries are partially markdown.
- We will use a reasoning model, as this is a reasoning task and we lack the scope for fine-tuning. *Ideally with larger scope, we'd fine tune a reasoning model.*
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

Then, I iteratively implemented a basic agent. I usually work with LangGraph, but considering the time-scope of this project, I chose to use the OpenAI Agents SDK for a (mostly) out of the box agent.

I equipped our agent with a set of tools, largely similar to those used in the original ConvFinQA paper with a couple of additions - ```percentage``` and ```direct_retrieval```. The ```direct_retrieval``` tool is used by the LLM to retrieve a figure directly from the context. I added this tool to ensure we uniformly process and output results, providing a direct logical path for the agent even for non-mathematic queries.

These tools output a ```ToolReturn``` object, which matches the format of the dataset to provide both a numeric answer and a program. 

This ```ToolReturn``` object is then used to generate the final output for the agent in the ```Answer``` model.

I then wrote some evaluation metrics, discussed below - one function per metric and a binding function to run all of them.

Once the key steps were complete - a parsed dataset, a working agent, and evaluation metrics - I wrote a script in main.py to bind all of these elements together to parse the dataset, run the agent, and compute evaluation metrics at the end of the run, for all the models specified. 

I also initially implemented some light containerization, but later scrapped it in favour of quickly making this code easily runnable with no further requirements or setup, while respecting the suggested time scope.

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

If I had more time and resources to experiment, I'd definitely test and evaluate multiple reasoning models and explore fine-tuning. The original paper notes that a neural symbolic approach was more successful in their original experiments; although it has been some years since the paper release, and model improvements have been significant, I'm curious to see what kind of performance we could achieve with a fine-tuning approach, notably reinforcement learning fine-tuning on a reasoning model like o4-mini.


#### Evaluation metric choices and reasoning

I selected the following evaluation metrics:

- Execution accuracy: was the answer right?
- Turn degradation: what is the average accuracy for each turn, and what is the degradation slope of accuracy over turns?
    

Execution accuracy is our key metric - getting the answer right is arguably the most important metric. Since our 'program' is programmatically generated via tool use, it's always equivalent if the answer is correct, so we don't require metrics for the program value.

Turn degradation is a metric I added to evaluate the agent's performance over time. It's a good indicator of whether the agent is deteriorating in performance over time. Turn degradation is a real problem for domain use: users expect consistent performance over time, and a drop in performance can lead to mistakes and trust degradation for our users.

If I had more time, I also would have:

- Dove into agent tool use: are there specific tools linked to high rates of failure? This is a whole-system evaluation: not just linked to model performance, but could be indicative of bugs in our system (like an incorrect mathematical tool or a description which confuses the model).
  
- Explored domain-specific error analysis to see what types of errors were most common - like confusing revenue and profits. I would do this with an evaluator LLM, analysing model responses to understand logic chains and identify knowledge gaps.

#### Results

Due to API cost, I ran evaluations on two models on just ten examples per model. As such, our results aren't at all indicative of overall performance - 10 examples is far too narrow a sample to measure real performance. 

GPT-4.1 performs better than o1-pro on this small dataset. This is interesting, and should be looked into further. However, to gauge real performance, a full test run is required.

Unfortunately, turn-based degradation is immeasurable in these results, due to a lack of adequate multi-turn samples being present in the sample. For example, our run on o1-pro featured at least one conversation with five turns - this single conversation makes up more than half of our results for the model. 

With a small sample, the varying factors which can surround failure can be easily conflated or missed. Identifying failure patterns in an accurate way, representative of the entire system, requires more data.

**GPT-4.1**
```
Overall Answer Accuracy: 50.00%

Accuracy Per Turn:
- 0    0.750000
- 1    0.666667
- 2    0.000000
- 5    0.000000
  
Turn Degradation Rate: -15.48%
```

**o1-pro**
```
Overall Answer Accuracy: 20.00%
Turn-based Accuracy:
- 0    1.0
- 1    0.0
- 2    0.0
- 3    0.0
- 4    0.0
- 5    0.0
Turn Degradation Rate: -14.29%
```

#### Strengths

The agent is simple and easy to use, with overall good latency and cost-effectiveness. The code is modular and ready to iterate on for a more robust, user-facing system. 

It's simple to run experiments of varying kinds with the ```mode``` parameter, which allows for easy switching between train/test/validation sets and a ```tiny``` option for a quick run. Likewise, it's simple to switch models by updating the ```llm``` parameter for our agent.

The data structure is comprehensive and clear, which is hugely beneficial for building a more robust system or exploring fine-tuning approaches in future. Having control over micro-elements of the data to construct well-formed prompts is a strength of this system.

#### Weaknesses and limitations

I ran out of time before I could finish everything I wanted to do, so the system definitely has some limitations. 

There is no true interactive entrypoint - our main script is only for running and reproducing evaluations, not for interactive use. If I had more time to iterate on this, I would have added a terminal-based entrypoint for interactive use with a random dataset sample, so the user can see the data, ask questions, and receive answers from the agent in a real multi-turn conversation.

Due to cost and time constraints, I was only able to run evaluations on ```tiny``` mode, which runs over 10 examples per model. I ran this tiny evaluation for all models I selected. From this small sample, the overall performance is not particularly good, and it's hard to identify turn-based accuracy on such a small set which may not feature a large number of multi-turn conversations, as discussed above.

With such a simple system, performance is largely reliant on the model we use. It's not a fully comprehensive system in that respect - just a workflow for handling multi-turn conversations based on the ConvFinQA dataset. If I had more time, not only would I evaluate multiple models for best performance, but I would add complexity and robustness to the agent for improved output.

I would constrain tool calling by setting a custom tool use behaviour for our agent. This would help ensure the agent does not make redundant or overcomplicated tool calls, which can lead to wasted tokens, incorrect answers, and system errors. I would run conversation persistence between turns using previous conversation ID, for elegance and to maintain a clearer context window. To improve multi-turn performance, I would explore post-turn summarization in place of persistent chat history acting as memory. 

#### What I'd do differently
This is what I would do differently if I had more time, or if I carry on with this project in the future, in order of importance:

- Run evaluation on our full test dataset to gain a clear understanding of systemic strengths and weaknesses based on performance.
- Reimplement containerization alongside a quickstart route for simple local runs.
- Implement CI/CD, just a basic GitHub workflow for now. 
- Create a custom tool use behaviour for the agent to avoid infinite tool loops and overcomplicated answer chains.
- Improve error handling and logging to prevent data loss and overall improve system performance.
- Evaluate different reasoning models to identify the best choice for this system - possibly fine-tune a model on the ConvFinQA dataset, ideally via reinforcement learning fine-tuning on a reasoning model.
- Add functions to measure latency and tokens per turn in order to optimize our workflow performance and scalability.
- Create an entrypoint for user queries.
- Evaluate results for domain-specific error analysis and tool-calling behaviour to identify common points of failure.
- Add a more comprehensive, domain-specific toolset, possibly even connect to relevant MCP server(s), to provide more context and agent capabilities beyond that of the original paper in order to provide a more comprehensive user experience.
- Potentially, translate the system to a model-agnostic LangGraph implementation to avoid vendor lock-in.

from .utils import async_retry
from .models import ToolReturn, Answer
from typing import List, Dict
from agents import (
    Agent,
    Runner,
    function_tool,
)

@function_tool
def add(a: int, b: int) -> ToolReturn:
    """Add two numbers."""
    return ToolReturn(answer=a + b, program=f"add({a}, {b})")

@function_tool   
def subtract(a: int, b: int) -> ToolReturn:
    """Subtract two numbers."""
    return ToolReturn(answer=a - b, program=f"subtract({a}, {b})")

@function_tool  
def multiply(a: int, b: int) -> ToolReturn:
    """Multiply two numbers."""
    return ToolReturn(answer=a * b, program=f"multiply({a}, {b})")
    
@function_tool 
def divide(a: int, b: int) -> ToolReturn:
    """Divide two numbers."""
    return ToolReturn(answer=a / b, program=f"divide({a}, {b})")

@function_tool
def percentage(a: int | float, b: int | float) -> ToolReturn:
    """Return the percentage of two numbers."""
    return ToolReturn(answer=100 *(float(a) / float(b)), program=f"percentage({a}, {b})")

@function_tool
def exponential(a: int, b: int) -> ToolReturn:
    """Return the exponential of two numbers."""
    return ToolReturn(answer=a ** b, program=f"exponential({a}, {b})")

@function_tool  
def greater(a: float | int, b: float | int) -> ToolReturn:
    """Compare two numbers: returns True if a > b"""
    result = a > b
    return ToolReturn(answer=result, program=f"greater({a}, {b})")

@function_tool
def direct_retrieval(a: int | float) -> ToolReturn:
    """ Return the correct answer format for a direct data lookup, when you don't have a program to return."""
    return ToolReturn(answer=a, program=a)

@async_retry(retries=3, delay=2, backoff=2)
async def run_agent(msg_chain: List[Dict[str, str]]):
    agent = Agent(
            name="Financial Conversation QA Agent",
            instructions="You are a helpful agent. You analyze financial documents and tables to answer questions accurately. For each user question, you return both the numerical answer and the program/calculation you used to derive it (e.g., answer: 117.3, program: subtract(9362.2, 9244.9)). You do not return any other text. Return your tool call output nand the associated program precisely if it is correct.",
            tools=[add, subtract, multiply, divide, percentage, exponential, greater, direct_retrieval],
            output_type=Answer,
            model = "gpt-4.1-mini-2025-04-14"
        )
    print("-- Running agent --")
    result = await Runner.run(agent, msg_chain)
    qa_history = result.to_input_list() # TODO: Try out using previous_response_id param to preserve qa_history rather than iteratively storing it
    answer = result.final_output
    print("-- Agent run complete --")
    return qa_history, answer
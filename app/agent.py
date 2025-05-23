# Basic agent to test method
import asyncio
from typing import Any
from pydantic import BaseModel, Field
from agents import (
    Agent,
    ModelSettings,
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

random_test_msg_chain = [
  {
    "role": "system",
    "content": "You are a helpful financial analysis agent. You analyze financial documents and tables to answer questions accurately. For each math question, you return both the numerical answer and the program/calculation you used to derive it (e.g., answer: 117.3, program: subtract(9362.2, 9244.9)). You do not return any other text. Return your tool call output nand the associated program precisely if it is correct. \n\nContext: Republic Services, Inc. 2008 Annual Report - Pro forma financial information following Allied acquisition:\n\nRevenue: 2008: $9,362.2M, 2007: $9,244.9M\nIncome from continuing operations: 2008: $285.7M, 2007: $423.2M\nBasic earnings per share: 2008: $0.76, 2007: $1.10\nDiluted earnings per share: 2008: $0.75, 2007: $1.09"
  },
  {
    "role": "user", 
    "content": "what were revenues in 2008?"
  },
  {
    "role": "assistant",
    "content": "9362.2"
  },
  {
    "role": "user",
    "content": "what were they in 2007?"
  },
  {
    "role": "assistant", 
    "content": "9244.9"
  },
  {
    "role": "user",
    "content": "what was the net change?"
  }
]

async def agent_test():
    agent = Agent(
            name="Math agent",
            instructions="You are a helpful agent. For each math question, you return the answer and the program you used to calculate it e.g answer: 5, program: add(2, 3)",
            tools=[add, subtract, multiply, divide, percentage, exponential, greater, direct_retrieval],
            model_settings=ModelSettings(
            ),
        )

    result = await Runner.run(agent, random_test_msg_chain)
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(agent_test())
    
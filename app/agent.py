# Basic agent to test method
import asyncio 
from .models import ToolReturn, Answer
from typing import Any, List, Dict
from pydantic import BaseModel, Field
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

async def run_agent(msg_chain: List[Dict[str, str]]):
    agent = Agent(
            name="Financial Conversation QA Agent",
            instructions="You are a helpful agent. You analyze financial documents and tables to answer questions accurately. For each user question, you return both the numerical answer and the program/calculation you used to derive it (e.g., answer: 117.3, program: subtract(9362.2, 9244.9)). You do not return any other text. Return your tool call output nand the associated program precisely if it is correct.",
            tools=[add, subtract, multiply, divide, percentage, exponential, greater, direct_retrieval],
            output_type=Answer
            # model = "o3-mini-2025-01-31",
        )

    print("Running agent...")
    print("-- msg_chain --")
    print(msg_chain)
    result = await Runner.run(agent, msg_chain)
    print(result)
    qa_history = result.to_input_list() # TODO: Use previous_response_id param to preserve qa_history rather than iteratively storing it
    print(qa_history)
    answer = result.final_output
    print(answer)
    return qa_history, answer

if __name__ == "__main__":
    random_test_msg_chain = [{'role': 'system', 'content': "**Context:**\n['substantially all of the goodwill and other intangible assets recorded related to the acquisition of allied are not deductible for tax purposes .', 'pro forma information the consolidated financial statements presented for republic include the operating results of allied from the date of the acquisition .', 'the following pro forma information is presented assuming the merger had been completed as of january 1 , 2007 .', 'the unaudited pro forma information presented below has been prepared for illustrative purposes and is not intended to be indicative of the results of operations that would have actually occurred had the acquisition been consummated at the beginning of the periods presented or of future results of the combined operations ( in millions , except share and per share amounts ) .', 'year ended december 31 , year ended december 31 , ( unaudited ) ( unaudited ) .']\n\n**Table:**\n|  | year ended december 31 2008 ( unaudited ) | year ended december 31 2007 ( unaudited ) |\n| --- | --- | --- |\n| revenue | $ 9362.2 | $ 9244.9 |\n| income from continuing operations available to common stockholders | 285.7 | 423.2 |\n| basic earnings per share | .76 | 1.10 |\n| diluted earnings per share | .75 | 1.09 |\n\n**Further information:**\n['the above unaudited pro forma financial information includes adjustments for amortization of identifiable intangible assets , accretion of discounts to fair value associated with debt , environmental , self-insurance and other liabilities , accretion of capping , closure and post-closure obligations and amortization of the related assets , and provision for income taxes .', 'assets held for sale as a condition of the merger with allied in december 2008 , we reached a settlement with the doj requiring us to divest of certain operations serving fifteen metropolitan areas including los angeles , ca ; san francisco , ca ; denver , co ; atlanta , ga ; northwestern indiana ; lexington , ky ; flint , mi ; cape girardeau , mo ; charlotte , nc ; cleveland , oh ; philadelphia , pa ; greenville-spartanburg , sc ; and fort worth , houston and lubbock , tx .', 'the settlement requires us to divest 87 commercial waste collection routes , nine landfills and ten transfer stations , together with ancillary assets and , in three cases , access to landfill disposal capacity .', 'we have classified the assets and liabilities we expect to divest ( including accounts receivable , property and equipment , goodwill , and accrued landfill and environmental costs ) as assets held for sale in our consolidated balance sheet at december 31 , 2008 .', 'the assets held for sale related to operations that were republic 2019s prior to the merger with allied have been adjusted to the lower of their carrying amounts or estimated fair values less costs to sell , which resulted in us recognizing an asset impairment loss of $ 6.1 million in our consolidated statement of income for the year ended december 31 , 2008 .', 'the assets held for sale related to operations that were allied 2019s prior to the merger are recorded at their estimated fair values in our consolidated balance sheet as of december 31 , 2008 in accordance with the purchase method of accounting .', 'in february 2009 , we entered into an agreement to divest certain assets to waste connections , inc .', 'the assets covered by the agreement include six municipal solid waste landfills , six collection operations and three transfer stations across the following seven markets : los angeles , ca ; denver , co ; houston , tx ; lubbock , tx ; greenville-spartanburg , sc ; charlotte , nc ; and flint , mi .', 'the transaction with waste connections is subject to closing conditions regarding due diligence , regulatory approval and other customary matters .', 'closing is expected to occur in the second quarter of 2009 .', 'republic services , inc .', 'and subsidiaries notes to consolidated financial statements %%transmsg*** transmitting job : p14076 pcn : 106000000 ***%%pcmsg|104 |00046|yes|no|02/28/2009 21:07|0|0|page is valid , no graphics -- color : d| .']"}, {'role': 'user', 'content': 'what were revenues in 2008?'}]
    asyncio.run(run_agent(random_test_msg_chain))
    
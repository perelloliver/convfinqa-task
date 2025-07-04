from typing import Any, Optional, List, Dict, Union
from pydantic import BaseModel, Field

class Answer(BaseModel):
    """Model for agent answer"""
    answer: Union[int, float]
    program: str

class ToolReturn(BaseModel):
    """Model for structured output from tools"""
    answer: Union[int, float] = Field(description="The answer to the question calculated by the tool")
    program: str = Field(description="The program used to answer the question e.g add(1, 2)")

class Turn(BaseModel):
    """Data model to save results and identifiers of each turn"""
    id: str = Field(description="conversation id")
    turn_index: int = Field(description="turn count")
    type: str = Field(description="Type I or II")
    qa_history: Optional[List] = Field(description="QA history of user<> question, system<> answer pairs")
    agent_answer: Optional[Any] = Field(description="answer from agent", default=None)
    agent_program: Optional[Any] = Field(description="program from agent", default=None)

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
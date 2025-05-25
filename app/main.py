from .agent import run_agent
from .models import Turn
from .parsing import parse_split_return_df
from .evals import run_eval
from .utils import filter_errors_for_eval, batch_data, flatten_turns
import pandas as pd
import asyncio
import argparse
from dotenv import load_dotenv

load_dotenv()

async def main(mode="tiny", llm="gpt-4.1-2025-04-14"):
    # Format the data, save for later use and return the df we're working with (test set for evals, tiny for code reproducibility)
    print(f"-- Formatting data for mode: {mode} --")
    test_data = parse_split_return_df("data/original_convfinqa_train.json", "data/formatted_dataset", mode)

    # Get conversation ids and iteratively run turns through the agent, preserving qa history for each turn
    # Save the answers to a single csv which holds all conversations, without overwriting previous conversation

    conversation_ids = test_data["id"].unique()
    conversation_batches = batch_data(conversation_ids)
    conversations = []

    print(f"-- Running agent with LLM: {llm} --")
    for batch_idx, batch in enumerate(conversation_batches):
        for id in batch:
            print(f"--conversation {id} start--")
            conversation = test_data[test_data["id"] == id]
            turns = []
            msg_chain = []
            msg_chain.append({"role": "user", "content": f"Review this financial data to answer my question: {conversation['context'].iloc[0]}"})
            
            try:
                for idx, turn in conversation.iterrows():
                    msg_chain.append({"role": "user", "content": turn["current_question"]})
                    qa_history, response = await run_agent(msg_chain, llm)
                    turn = Turn(
                        id=id,
                        turn_index=turn["turn_index"],
                        type=turn["type"],
                        qa_history=msg_chain,
                        agent_answer=response.answer,
                        agent_program=response.program
                    )
                    msg_chain = qa_history
                    turns.append(turn)
            except Exception as e:
                print(f"Error for id={id}: {type(e).__name__}: {e}")
                turns.append({
                    'id': id,
                    'turn_index': None,
                    'type': None,
                    'qa_history': None,
                    'agent_answer': None,
                    'agent_program': None,
                    'error': True,
                    'error_type': type(e).__name__
                })
            except Exception as e:
                print(f"Error for id={id}: {type(e).__name__}: {e}")
                turns.append({
                    'id': id,
                    'turn_index': None,
                    'type': None,
                    'qa_history': None,
                    'agent_answer': None,
                    'agent_program': None,
                    'error': True,
                    'error_type': type(e).__name__
                })
            print(f"--conversation {id} complete--")
            conversations.append(turns)

    results_df = flatten_turns(conversations)
    results_df.to_csv("data/responses.csv", mode="a", index=False)

    filtered_results, filtered_test, n_errors, error_types = filter_errors_for_eval(results_df, test_data)
    print(f"Filtered out {n_errors} conversations with errors from evaluations. Error types encountered: {error_types}")
    print(f"-- RESULTS FOR MODEL {llm} --")
    run_eval(filtered_results, filtered_test)

async def run(mode:str, llms:list):
    tasks = [main(mode=mode, llm=llm) for llm in llms]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ConvFinQA agent and evaluation.")
    parser.add_argument('--mode', type=str, default="tiny", choices=["tiny", "test", "full"], help='Which data split to run: tiny (quick test) or test (full evals) or full (entire dataset performance)')
    args = parser.parse_args()
    asyncio.run(main(mode=args.mode, llm="gpt-4.1-mini-2025-04-14")) # Dev
    # asyncio.run(run(mode=args.mode, llms=["o1-pro-2025-03-19", "gpt-4.1-2025-04-14" ])) # Evals
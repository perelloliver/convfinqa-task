from .agent import run_agent
from .models import Turn
from data.parsing import parse_split_return_df
from .evals import run_eval
from .utils import batch_data, flatten_turns
import pandas as pd
import asyncio

import argparse

async def main(mode="tiny"):
    # Format the data, save for later use and return the df we're working with (test set for evals, tiny for code reproducibility)
    print(f"-- Formatting data for mode: {mode} --")
    test_data = parse_split_return_df("/Users/mac/convfinqa-task/data/original_convfinqa_train.json", "/Users/mac/convfinqa-task/data/formatted_dataset", mode)
    print("-- Data formatted --")

    # Get conversation ids and iteratively run turns through the agent, preserving qa history for each turn
    # Save the answers to a single csv which holds all conversations, without overwriting previous conversation

    conversation_ids = test_data["id"].unique()
    conversation_batches = batch_data(conversation_ids)
    conversations = []

    for batch_idx, batch in enumerate(conversation_batches):
        print(f"--batch {batch_idx} start--")
        for id in batch:
            print(f"--conversation {id} start--")
            conversation = test_data[test_data["id"] == id]
            turns = []
            msg_chain = []
            msg_chain.append({"role": "user", "content": f"Review this financial data to answer my question: {conversation['context'].iloc[0]}"})
            
            for idx, turn in conversation.iterrows():
                print(f"--turn {turn.turn_index} start--")
                msg_chain.append({"role": "user", "content": turn["current_question"]})
                qa_history, response = await run_agent(msg_chain)
                print("---DEBUG FORMATTING---")
                print(response)
                print(qa_history)
                turn = Turn(
                    id=id,
                    turn_index=turn["turn_index"],
                    type=turn["type"],
                    qa_history=msg_chain,
                    agent_answer=response.answer,
                    agent_program=response.program
                )
                print(f"--turn {turn.turn_index} complete--")
                print(turn)
                msg_chain = qa_history
                turns.append(turn)
            print(f"--conversation {id} complete--")
            conversations.append(turns)
        print(f"--batch {batch_idx} complete--")

    # Save to csv using helper for DRY
    results_df = flatten_turns(conversations)
    results_df.to_csv("/Users/mac/convfinqa-task/data/responses.csv", mode="a", index=False)

    # Debug: print columns before eval
    print("results_df columns:", results_df.columns)
    print("test_data columns:", test_data.columns)

    # Run eval metrics and print results
    print("-- Running Evaluation... --")
    run_eval(results_df, test_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ConvFinQA agent and evaluation.")
    parser.add_argument('--mode', type=str, default="tiny", choices=["tiny", "test", "full"], help='Which data split to run: tiny (quick test) or test (full evals) or full (entire dataset performance)')
    args = parser.parse_args()
    asyncio.run(main(mode=args.mode))
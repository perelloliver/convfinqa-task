from .agent import run_agent
from .models import Turn
from data.parsing import parse_split_return_df
import numpy as np
import asyncio

async def main():
    # Format the data, save for later use and return the df we're working with (test set in this case)
    print("-- Formatting data --")
    test_data = parse_split_return_df("/Users/mac/convfinqa-task/data/original_convfinqa_train.json", "/Users/mac/convfinqa-task/data/formatted_dataset", "test")
    print("-- Data formatted --")

    # Get conversation ids and iteratively run turns through the agent, preserving qa history for each turn
    # Save the answers to a single csv which holds all conversations, without overwriting previous conversation

    conversation_ids = test_data["id"].unique()
    conversation_batches = np.array_split(conversation_ids, 100)
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
                    id=turn["id"],
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

    # Save to csv
    with open("/Users/mac/convfinqa-task/data/responses.csv", "a") as f:
        results_df = pd.DataFrame(conversations)
        results_df.to_csv(f, index=False)

    # Run eval metrics and print results
    print("-- Running Evaluation... --")
    # evals = evaluate(results_df)
    # print(evals)

if __name__ == "__main__":
    asyncio.run(main())
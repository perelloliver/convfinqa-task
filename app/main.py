from .agent import run_agent
from .models import Turn
from .parsing import parse_split_return_df

# Format the data, save for later use and return our df
test_data = parse_split_return_df("/Users/mac/convfinqa-task/data/dataset/train.json", "/Users/mac/convfinqa-task/data/formatted_dataset", "test")

# Unpack df into conversations, iteratively build our message chain by turn and run the agent
# Agent returns answer and new cumulative qa history

# Get conversation ids and iteratively run them through the agent, preserving qa history for each turn
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
        msg_chain.append({"role": "system", "content": conversation["context"].iloc[0]})
        
        for turn in conversation.iterrows():
            print(f"--turn {turn['turn_index']} start--")
            msg_chain.append({"role": "user", "content": turn["current_question"]})
            response, qa_history = run_agent(msg_chain) # Append the qa history to the message chain
            turn = Turn(
                id=turn["id"],
                turn_index=turn["turn_index"],
                type=turn["type"],
                qa_history=msg_chain,
                agent_answer=response,
                agent_program=qa_history[-1]["program"]
            )
            print(f"--turn {turn['turn_index']} complete--")
            print(turn)
            msg_chain.extend(qa_history)
            turns.append(turn)
        print(f"--conversation {id} complete--")
        conversations.append(turns)
    print(f"--batch {batch_idx} complete--")

# Save to csv
# TODO: If time, add batching here.
with open("/Users/mac/convfinqa-task/data/responses.csv", "a") as f:
    results_df = pd.DataFrame(conversations)
    results_df.to_csv(f, index=False)

# Run eval metrics and print results

# evals = evaluate(results_df)
# print(evals)

from typing import Any, List, Dict
import pandas as pd
from app.models import DatasetConvQaParsed
from sklearn.model_selection import train_test_split
import os

def format_context(pre_text: str, table: Any, post_text: str) -> str:
    """
    Formats the context as a markdown string using pre_text, table, and post_text.
    Table is assumed to be a list of lists or a markdown string.
    """
    md = ''
    if pre_text:
        md += f"**Context:**\n{pre_text}\n\n"
    if table:
        if isinstance(table, str):
            md += f"**Table:**\n{table}\n\n"
        elif isinstance(table, list):
            # Convert list of lists to markdown table
            if table and isinstance(table[0], list):
                header = '| ' + ' | '.join(str(h) for h in table[0]) + ' |\n'
                separator = '| ' + ' | '.join(['---'] * len(table[0])) + ' |\n'
                rows = ''.join(['| ' + ' | '.join(str(cell) for cell in row) + ' |\n' for row in table[1:]])
                md += f"**Table:**\n{header}{separator}{rows}\n"
            else:
                md += f"**Table:**\n{table}\n\n"
        else:
            md += f"**Table:**\n{str(table)}\n\n"
    if post_text:
        md += f"**Further information:**\n{post_text}"
    return md.strip()


def build_qa_history(dialogue_break: List[str], exe_ans_list: List[str], turn_idx: int) -> List[Dict[str, str]]:
    """
    Build QA history up to the current turn (exclusive).
    """
    return [
        {"user": str(question), "system": str(answer)}
        for question, answer in zip(dialogue_break[:turn_idx], exe_ans_list[:turn_idx])
    ]


def parse_conversation_entry(entry: Dict) -> List[DatasetConvQaParsed]:
    """
    Parse a single conversation entry into a list of DatasetConvQaParsed objects (one per turn).
    """
    annotation = entry["annotation"]
    pre_text = entry.get("pre_text", "")
    table = entry.get("table", "")
    post_text = entry.get("post_text", "")
    context = format_context(pre_text, table, post_text)
    dialogue_break = annotation["dialogue_break"]
    exe_ans_list = annotation["exe_ans_list"]
    turn_program = annotation["turn_program"]
    qa_split = annotation.get("qa_split", [0] * len(dialogue_break))
    conv_type = "Type I" if all(x == 0 for x in qa_split) else "Type II"
    parsed_items = []
    for turn_idx, question in enumerate(dialogue_break):
        print(f"Parsing turn {turn_idx} of {len(dialogue_break)}")
        qa_history = build_qa_history(dialogue_break, exe_ans_list, turn_idx)
        current_question = question
        gold_program = turn_program[turn_idx]
        gold_answer = exe_ans_list[turn_idx]
        parsed = DatasetConvQaParsed(
            context=context,
            qa_history=qa_history,
            current_question=current_question,
            gold_program=gold_program,
            gold_answer=gold_answer,
            id=entry["id"],
            turn_index=str(turn_idx),
            type=conv_type
        )
        parsed_items.append(parsed)
    print(f"Data parsing complete for {len(parsed_items)} entries.")
    return parsed_items


def flatten_and_dictify(parsed_data: List[List[DatasetConvQaParsed]]) -> List[Dict]:
    """
    Flatten a list of lists of DatasetConvQaParsed objects and convert to list of dicts.
    """
    flat = [item for sublist in parsed_data for item in sublist]
    return [obj.model_dump() for obj in flat]


def parsed_to_dataframe(parsed_data: List[List[DatasetConvQaParsed]]) -> pd.DataFrame:
    """
    Convert parsed data (list of lists of DatasetConvQaParsed) directly to a DataFrame.
    """
    dicts = flatten_and_dictify(parsed_data)
    return pd.DataFrame(dicts)


def dataset_parse(json_path: str) -> pd.DataFrame:
    """
    Load a dataset JSON file, parse it, and return a DataFrame with one row per turn.
    """
    import json
    with open(json_path, 'r') as f:
        data = json.load(f)
    parsed = [parse_conversation_entry(entry) for entry in data]
    return parsed_to_dataframe(parsed)


def dataset_split_preserve_conversations(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Split our dataset while preserving conversation integrity, keeping all turns of a conversation in the same split.
    """
    conv_info = df[['id', 'type']].drop_duplicates()

    train_ids, temp_ids = train_test_split(
        conv_info,
        test_size=0.3,
        random_state=42,
        stratify=conv_info['type']
    )

    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.5,
        random_state=42,
        stratify=temp_ids['type']
    )

    # Assign all rows of an ID to the assigned split
    train = df[df['id'].isin(train_ids['id'])].reset_index(drop=True)
    val = df[df['id'].isin(val_ids['id'])].reset_index(drop=True)
    test = df[df['id'].isin(test_ids['id'])].reset_index(drop=True)

    # Assert there is no overlap (bugcheck just in case I missed something in the logic)
    assert set(train['id']) & set(val['id']) == set()
    assert set(train['id']) & set(test['id']) == set()
    assert set(val['id']) & set(test['id']) == set()
    print("--- No conversation ID overlap in train, test, or validation sets detected ---")

    # Save to dir provided
    train.to_csv(output_dir + "/train_70.csv", index=False)
    val.to_csv(output_dir + "/val_15.csv", index=False)
    test.to_csv(output_dir + "/test_15.csv", index=False)
    return train, test, val
    
def parse_split_return_df(input_json: str, output_dir: str, return_df: str = "full") -> pd.DataFrame:
    """
    Parses the dataset, saves splits, and returns the requested DataFrame (default full).
    return_df: one of 'train', 'test', 'val', 'full'
    """
    os.makedirs(output_dir, exist_ok=True)
    formatted_data = dataset_parse(input_json)
    print(f"Data formatting complete for {len(formatted_data)} entries.")
    print("--- DATA SAMPLE --- ")
    print(formatted_data[:2])
    formatted_data.to_csv(os.path.join(output_dir, "train_no_split.csv"), index=False)
    train, test, val = dataset_split_preserve_conversations(formatted_data, output_dir)
    print("Data splitting complete, all files saved.")

    df_map = {
        "train": train,
        "test": test,
        "val": val,
        "full": formatted_data
    }
    try:
        return df_map[return_df]
    except KeyError:
        raise ValueError(f"Unknown return_df value: {return_df}. Choose from {list(df_map.keys())}.")


    

import pandas as pd
import numpy as np

# Evaluation metrics implementation

def answer_accuracy(df):
    """Compute answer accuracy (exact match or close for floats)"""
    def is_correct(row):
        a, g = row['agent_answer'], row['gold_answer']
        try:
            # Try float comparison with tolerance
            return np.isclose(float(a), float(g), atol=1e-3)
        except Exception:
            return str(a).strip() == str(g).strip()
    correct = df.apply(is_correct, axis=1)
    return correct.mean(), correct

def turn_based_performance(df):
    """Compute accuracy per turn_index and overall"""
    turn_acc = df.groupby('turn_index').apply(lambda g: (g['agent_answer'] == g['gold_answer']).mean())
    overall = (df['agent_answer'] == df['gold_answer']).mean()
    return turn_acc, overall

def run_eval(orig_df, agent_df):
    """
    Evaluate agent outputs against gold data.
    Args:
        orig_df (pd.DataFrame): Original dataset dataframe (must include gold_answer, id, turn_index, type, etc)
        agent_df (pd.DataFrame): Agent output dataframe (must include agent_answer, id, turn_index, type, etc)
    """
    # Merge on id and turn_index
    merged = pd.merge(orig_df, agent_df, on=['id', 'turn_index', 'type'], suffixes=('', '_agent'))
    # Map agent columns to expected names if needed
    for col in ['agent_answer', 'agent_program']:
        if col not in merged.columns:
            merged[col] = merged.get(f'{col}_agent', None)
    print("--- Evaluation Results ---")
    acc, _ = answer_accuracy(merged)
    print(f"Answer Accuracy: {acc:.3f}")
    turn_acc, overall = turn_based_performance(merged)
    print("Turn-based Accuracy:")
    print(turn_acc)
    print(f"Overall Turn-based Accuracy: {overall:.3f}")
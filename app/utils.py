import numpy as np
import asyncio 
import time
from functools import wraps

def to_percent(val, decimals=2):
    """Convert a float (0-1) to a percentage string with specified decimals."""
    return f"{val * 100:.{decimals}f}%"

def batch_data(data):
    """Split data into batches of 100 for handling large datasets"""
    num_samples = len(data)
    split_count = min(100, num_samples)
    return np.array_split(data, split_count)

def flatten_turns(conversations):
    """
    Flatten a list of lists of Turn objects and return as DataFrame.
    Each Turn should be a Pydantic model or have a .dict() method.
    """
    import pandas as pd
    flat_turns = [t.dict() if hasattr(t, "dict") else vars(t) for turns in conversations for t in turns]
    df = pd.DataFrame(flat_turns)
    return df

def async_retry(retries=3, delay=1, backoff=2):
    """Async retry decorator with exponential backoff.
    Handle rate-limiting and/or other exceptions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            m_delay = delay
            for attempt in range(1, retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    print(f"[Retry {attempt}/{retries}] Error: {e}")
                    if attempt == retries:
                        print("Max retries reached. Raising error.")
                        raise
                    await asyncio.sleep(m_delay)
                    m_delay *= backoff
        return wrapper
    return decorator
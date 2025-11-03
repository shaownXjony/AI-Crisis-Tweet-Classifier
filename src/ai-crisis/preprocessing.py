import re
from typing import Optional

def simple_clean(text: Optional[str]) -> str:
    """
    Clean tweet text for classification.
    - Lowercases everything
    - Removes URLs, mentions, hashtags, and RT
    - Keeps letters, numbers, and spaces
    """
    if not isinstance(text, str):
        return ""
    
    s = text.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)  # remove URLs
    s = re.sub(r"@\w+", " ", s)              # remove @mentions
    s = re.sub(r"#", " ", s)                 # remove #
    s = re.sub(r"\brt\b", " ", s)            # remove "RT"
    s = re.sub(r"[^a-z0-9\s']", " ", s)      # remove special characters
    s = re.sub(r"\s+", " ", s).strip()       # remove extra spaces
    
    return s

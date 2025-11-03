from src.ai_crisis.preprocessing import simple_clean

def test_simple_clean_none():
    assert simple_clean(None) == ""

def test_simple_clean_basic():
    """Remove URL, mention, RT and punctuation, and lowercase."""
    s = "RT @user: Help! Flood at http://example.com #flooding"
    cleaned = simple_clean(s)
    assert "http" not in cleaned
    assert "@" not in cleaned
    assert "rt" not in cleaned  # rt token removed
    assert "flooding" in cleaned
    assert cleaned == cleaned.lower()

def test_simple_clean_apostrophe_keep():
    """Keep contractions/apostrophes (e.g. don't)."""
    s = "Don't panic! #help"
    cleaned = simple_clean(s)
    assert "don't" in cleaned or "dont" in cleaned

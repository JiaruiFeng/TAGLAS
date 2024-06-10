import re
import string

def normalize_text(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    text = text.lower()
    exclude = set(string.punctuation)
    # remove <pad> token:
    text = re.sub(r"<pad>|</s>|<s>|<bos>|<eos>", "", text)
    text = "".join(char for char in text if char not in exclude)
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    text = " ".join(text.split())
    return text



def extract_numbers(text:str, default_value: float = 0.0):
    """ Extracts all numbers from a given text and returns them as a list of floats. """
    numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", text)
    if len(numbers) == 0:
        # default value
        return [default_value]
    return [float(num) for num in numbers]


def pattern_exist(text: str, pattern: str, exact_match: bool = True, sep: str = " "):
    if exact_match:
        return pattern in text
    else:
        texts = text.split(sep)
        patterns = pattern.split(sep)
        for pattern in patterns:
            if pattern not in texts:
                return False
        return True

def pattern_search(text: str, pattern: str):
    m = re.findall(pattern, text)
    return m

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Salesforce/SFR-Embedding-Code-400M_R",
    trust_remote_code=True
)

def count_tokens(text: str, add_special: bool = False) -> int:
    """
    Trả về số token theo tokenizer của model embedding.
    
    add_special: nếu True thì bao gồm các token đặc biệt (CLS, SEP, ...)
    """
    tokens = tokenizer.encode(
        text,
        add_special_tokens=add_special,
        truncation=False,
        max_length=None
    )
    return len(tokens)

def load_json(json_file):
    with open(json_file) as f:
        return json.load(f)

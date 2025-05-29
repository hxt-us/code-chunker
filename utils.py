from transformers import AutoTokenizer

def count_tokens(string: str, encoding_name: str, add_special: bool = False) -> int:

    tokenizer = AutoTokenizer.from_pretrained(
    "Salesforce/SFR-Embedding-Code-400M_R",
    trust_remote_code=True
)
    tokens = tokenizer.encode(
        string,
        add_special_tokens=add_special,
        truncation=False,
        max_length=None
    )
    return len(tokens)

def load_json(json_file):
    with open(json_file) as f:
        return json.load(f)

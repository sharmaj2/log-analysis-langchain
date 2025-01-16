from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

query = "Which IP addresses made A-type queries to usma.bluenet on November 7, 2011?"
token_count = len(tokenizer.encode(query))

print(f"Query token count: {token_count}")
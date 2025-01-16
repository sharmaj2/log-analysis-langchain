from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Get the Hugging Face access token from the environment
hf_access_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
if not hf_access_token:
    raise ValueError("Access token not found in the environment. Please check .env file.")

# Load tokenizer and model
llm_model_name = "meta-llama/Llama-2-7b-chat-hf"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name, access_token=hf_access_token)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_name, 
    device_map="cuda", 
    torch_dtype="float16",
)

llm_pipeline = pipeline("text-generation", model=llm_model, tokenizer=llm_tokenizer)
print(f"Model '{llm_model_name}' loaded successfully!")


# Define a test prompt
prompt = "Why there is French inflence in Canada?"

# Generate a response
response = llm_pipeline(prompt, max_length=100, num_return_sequences=1, truncation=True)

# Print the output
print("\nResponse:")
print(type(response))
print(response)
print(response[0]['generated_text'])
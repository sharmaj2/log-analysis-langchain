from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

# Set up Pinecone API
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("Access token not found in the environment. Please check .env file.")

pc = Pinecone(api_key=pinecone_api_key)
index_name = "dns-log-index"

# Check if the index exists; if not, create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# Connect to the Pinecone index
index = pc.Index(index_name)
print(f"Pinecone index '{index_name}' is ready.")

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': False},
    show_progress=True
)

# Set up Pinecone vector store
vector_store = PineconeVectorStore(
    index=index,
    embedding=embedding_model
)
print("Pinecone vector store initialized.")

# Load the DNS log file using LangChain's TextLoader
log_file_path = "./dataset/dns_log_file.txt"
loader = TextLoader(log_file_path)
documents = loader.load()

# Split the logs into individual entries
newline_splitter = RecursiveCharacterTextSplitter(separators=["\n"], chunk_size=1, chunk_overlap=0)
line_split_documents = newline_splitter.split_documents(documents)

# Prepare documents for adding to Pinecone
documents = [
    Document(page_content=doc.page_content, metadata=doc.metadata)
    for doc in line_split_documents
]
ids = [str(i) for i in range(len(documents))]

# print(len(documents))

# Check existing documents in Pinecone and add new ones
index_stats = vector_store._index.describe_index_stats()

# print(index_stats)

if index_stats["total_vector_count"] == len(documents):
    print("No documents to add to Pinecone vector store.")
else:
    vector_store.add_documents(documents=documents, ids=ids)
    print("New documents added to Pinecone vector store.")


# Get the Hugging Face access token from the environment
hf_access_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
if not hf_access_token:
    raise ValueError("Hugging Face access token not found in the environment. Please check .env file.")

# Load tokenizer and model
llm_model_name = "meta-llama/Llama-2-7b-chat-hf"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name, access_token=hf_access_token)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    device_map="cuda",
    torch_dtype="float16"
)

# Initialize the Llama-2 model pipeline
llm_pipeline = pipeline("text-generation", model=llm_model, tokenizer=llm_tokenizer)
print(f"Model '{llm_model_name}' loaded successfully!")

# Wrap the Hugging Face pipeline in LangChain's LLM
llm = HuggingFacePipeline(pipeline=llm_pipeline)
print("Model wrapped in LangChain successfully.")

# Using MMR search
retriever = vector_store.as_retriever( search_type="mmr", search_kwargs={"k": 5, "fetch_k": 100, "lambda_mult": 0.2})

# # Using similarity_search_with_score
# retriever = vector_store.as_retriever(
#     search_type="similarity", 
#     search_kwargs={"k": 5}
# )

# Set up RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Define a sample query
query = "Which IP addresses queried jabber.usma.bluenet?"

# Execute the query and display the response
response = qa_chain.invoke({"query": query})
print("\nAnswer:")
print(response["result"])

print("\nSource Documents:")
for i, doc in enumerate(response["source_documents"]):
    print(f"Document {i + 1}: {doc.page_content}")
    print("-" * 50)

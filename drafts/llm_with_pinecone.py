from pinecone import Pinecone, ServerlessSpec

from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Get the Pinecone API key from the environment
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("Access token not found in the environment. Please check .env file.")

pc = Pinecone(api_key=pinecone_api_key)

# Create a Pinecone index
index_name = "dns-log-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ) 
)

# Connect to the index
index = pc.Index(index_name)
print(f"Pinecone index '{index_name}' is ready.")

from langchain_huggingface import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    show_progress=True

)

from langchain_pinecone import PineconeVectorStore

# Create Pinecone vector store
vector_store = PineconeVectorStore(
    index=index,
    embedding=embedding_model
)

print("Pinecone vector store initialized.")


from langchain_community.document_loaders import TextLoader

# # Load the log file using LangChain's TextLoader
# log_file_path = "./dataset/dns_log_file.txt"
# loader = TextLoader(log_file_path)

# # Load the documents
# documents = loader.load()

# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # Initialize the splitter to split by new line
# newline_splitter = RecursiveCharacterTextSplitter(separators=["\n"], chunk_size=1, chunk_overlap=0)

# # Split the loaded documents by new line
# line_split_documents = newline_splitter.split_documents(documents)


# from langchain.docstore.document import Document

# documents = [
#     Document(page_content=doc.page_content, metadata=doc.metadata)
#     for doc in line_split_documents
# ]
# ids = [str(i) for i in range(len(documents))]

# # Add documents to the vector store
# vectordb.add_documents(documents=documents, ids=ids)
# print("Documents added to Pinecone vector store.")

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

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


from langchain_huggingface import HuggingFacePipeline

# Wrap the Hugging Face pipeline in LangChain's LLM
llm = HuggingFacePipeline(pipeline=llm_pipeline)
print("Model wrapped in LangChain successfully.")


retriever = vector_store.as_retriever( search_type="mmr", search_kwargs={"k": 5, "fetch_k": 50, "lambda_mult": 0.5})

from langchain.chains import RetrievalQA

# # Set up RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Define a simple query
query = "how many times failed to connect: timed out occurred?"

# Query the QA chain
response = qa_chain.invoke({"query": query}, filter={"text":"time out"})

# response = retriever.invoke(query)

# print(res)
# Display the answer
print("\nAnswer:")
print(response["result"])

# Display source documents
print("\nSource Documents:")
for i, doc in enumerate(response["source_documents"]):
    print(f"Document {i + 1}: {doc.page_content}")
    print("-" * 50)
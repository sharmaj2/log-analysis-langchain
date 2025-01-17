from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

from dotenv import load_dotenv
import os
from uuid import uuid4

# Load environment variables from the .env file
load_dotenv()

# Get the Pinecone API key from the environment
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError(
        "Access token not found in the environment. Please check .env file.")

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


model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    show_progress=True

)


# Create Pinecone vector store
vector_store = PineconeVectorStore(
    index=index,
    embedding=embedding_model, namespace="logs-with-metadata"
)

print("Pinecone vector store initialized.")

# Load the log file using LangChain's TextLoader
log_file_path = "./dataset/dns_log_file.txt"
loader = TextLoader(log_file_path)

# Load the documents
documents = loader.load()


# Initialize the splitter to split by new line
# splitter = RecursiveCharacterTextSplitter(separators=["\n", " "], chunk_size=11, chunk_overlap=3)
splitter = CharacterTextSplitter(separator="\n", chunk_size=200, chunk_overlap=20)


# Split the loaded documents by new line
split_docs = splitter.split_documents(documents)


# # Prepare documents for adding to Pinecone
# documents = [
#     Document(page_content=doc.page_content, metadata={"line": doc.page_content})
#     for doc in split_docs
# ]

# uuids = [str(uuid4()) for _ in range(len(documents))]

# vector_store.add_documents(documents=documents, ids=uuids)

# # Query the vector store
# query = "how many quries made on Nob 8 2011?"
# vector = embedding_model.embed_query(query)

# # Search in Pinecone
# results = vector_store.similarity_search_with_score(query, k=5)

# # Display results
# for match, score in results:
#     print(f"Log Line: {match.page_content}")
#     print(f"Metadata: {match.metadata}")
#     print(f"Score: {score}")
#     print("-" * 40)


# Get the Hugging Face access token from the environment
hf_access_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
if not hf_access_token:
    raise ValueError(
        "Access token not found in the environment. Please check .env file.")

# Load tokenizer and model
llm_model_name = "meta-llama/Llama-2-7b-chat-hf"
llm_tokenizer = AutoTokenizer.from_pretrained(
    llm_model_name, access_token=hf_access_token)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    device_map="cuda",
    torch_dtype="float16",
)

llm_pipeline = pipeline(
    "text-generation", model=llm_model, tokenizer=llm_tokenizer)
print(f"Model '{llm_model_name}' loaded successfully!")


# Wrap the Hugging Face pipeline in LangChain's LLM
llm = HuggingFacePipeline(pipeline=llm_pipeline)
print("Model wrapped in LangChain successfully.")

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Wrap our vectorstore
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_store.as_retriever()
)


# # Set up RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=compression_retriever,
    return_source_documents=True
)

# Define a simple query
query = "how many times failed to connect: timed out occurred?"

# Query the QA chain
response = qa_chain.invoke({"query": query}, filter={"line": "time out"})

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

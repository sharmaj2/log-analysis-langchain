from langchain.vectorstores import FAISS
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

persist_directory = "./faiss_store"

# Reload the FAISS vector store
vectordb = FAISS.load_local(persist_directory, embeddings=embedding_model,  allow_dangerous_deserialization=True)

# # Test retrieval
# query = "Which IP addresses made queries to usma.bluenet?"
# retrieved_docs = vectordb.similarity_search(query, k=3)

# # Display results
# for i, doc in enumerate(retrieved_docs):
#     print(f"Document {i + 1}: {doc.page_content}")
#     print("-" * 50)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load the Flan-T5 model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="cuda")

# Create a text2text generation pipeline
llm_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

print(f"Model '{model_name}' loaded successfully!")


from langchain_huggingface import HuggingFacePipeline

# Wrap the Hugging Face pipeline in LangChain's LLM
llm = HuggingFacePipeline(pipeline=llm_pipeline)
print("Model wrapped in LangChain successfully.")


# from langchain.chains import RetrievalQA

# # Set up a RetrievalQA chain
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,  # Use the Flan-T5 model
#     retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 10}),  # Use FAISS as the retriever
#     return_source_documents=True  # Return retrieved documents
# )

# print("RetrievalQA chain set up successfully!")

# # Define a query
# query = "List the IP addresses that made DNS queries to the domain usma.bluenet on November 7, 2011."

# # Get the response
# # response = qa_chain({"query": query})
# response = qa_chain.invoke({"query":query})

# # Display the answer
# print("\nAnswer:")
# print(response["result"])

# # Display source documents
# print("\nSource Documents:")
# for i, doc in enumerate(response["source_documents"]):
#     print(f"Document {i + 1}: {doc.page_content}")
#     print("-" * 50)

from langchain.chains.query_constructor.base import AttributeInfo

# Define metadata fields for filtering
metadata_field_info = [
    AttributeInfo(
        name="date",
        description="The timestamp of the log entry",
        type="string",
    ),
    AttributeInfo(
        name="client_ip",
        description="The IP address of the client making the query",
        type="string",
    ),
    AttributeInfo(
        name="query_type",
        description="The type of query in the DNS log (e.g., A, PTR, MX)",
        type="string",
    ),
]

from langchain.prompts import PromptTemplate
from langchain.retrievers.self_query.base import SelfQueryRetriever

# Define a custom prompt template for query reformulation
self_query_prompt = PromptTemplate.from_template(
    "Given the following metadata fields: {metadata_field_info}, reformulate the user's query to retrieve the most relevant documents."
)

# Initialize the SelfQueryRetriever
retriever = SelfQueryRetriever.from_llm(
    llm=llm,  # Use your Flan-T5 model wrapped as HuggingFacePipeline
    vectorstore=vectordb,  # Your FAISS vector store
    document_contents="A DNS log entry with metadata",
    metadata_field_info=metadata_field_info,
    prompt_template=self_query_prompt,
)

print("SelfQueryRetriever initialized successfully!")


# Define a query
query = "Which IP addresses made A-type queries to usma.bluenet on November 7, 2011?"

# Retrieve documents
retrieved_docs = retriever.get_relevant_documents(query)

# Display results
print("\nRetrieved Documents:")
for i, doc in enumerate(retrieved_docs):
    print(f"Document {i + 1}: {doc.page_content}")
    print("-" * 50)

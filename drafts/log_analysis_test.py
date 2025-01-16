from langchain_community.document_loaders import TextLoader

# Load the log file using LangChain's TextLoader
log_file_path = "./dns_log_file.txt"
loader = TextLoader(log_file_path)

# Load the documents
documents = loader.load()

# # Verify the first few documents
# print(f"Number of documents loaded: {len(documents)}")
# for i, doc in enumerate(documents[:5]):
#     print(f"Document {i + 1} Content:")
#     print(doc.page_content)
#     print("-" * 50)

# print(documents[0].metadata)


from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize the splitter to split by new line
newline_splitter = RecursiveCharacterTextSplitter(separators=["\n"], chunk_size=1, chunk_overlap=0)

# Split the loaded documents by new line
line_split_documents = newline_splitter.split_documents(documents)

document_texts = [doc.page_content for doc in line_split_documents]

from langchain.docstore.document import Document

# Convert split documents back to LangChain Documents
documents = [Document(page_content=text) for text in document_texts]

print(type(documents))

# # Display Results
# print(f"Total Documents Split by Line: {len(line_split_documents)}")

# # Preview the first 5 split documents
# for i, doc in enumerate(line_split_documents[:5]):
#     print(f"Document {i + 1}:")
#     print(doc.page_content)
#     print("-" * 50)


# persist_directory = "./faiss_store"

# from tqdm import tqdm
# from langchain_huggingface import HuggingFaceEmbeddings

# # embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/e5-large-v2')

# # Load e5-large-v2 with SentenceTransformer
# # embedding_model = SentenceTransformer("intfloat/e5-large-v2")

# # embedding_model = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')
# model_name = "sentence-transformers/all-mpnet-base-v2"
# model_kwargs = {'device': 'cuda'}
# encode_kwargs = {'normalize_embeddings': False}
# embedding_model = HuggingFaceEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs,
#     show_progress=True

# )

# # # Define a custom embedding function for LangChain
# # def embedding_function(texts):
# #     return embedding_model.encode(texts, show_progress_bar=True)

# print("all-mpnet-base-v2 model loaded successfully!")


# from langchain_community.vectorstores import FAISS
# from langchain.docstore.document import Document


# vectordb = FAISS.from_documents(
#     documents=line_split_documents,
#     embedding=embedding_model
# )

# FAISS.lo

# vectordb.save_local(persist_directory)
# print(f"FAISS vector store saved at {persist_directory}.")


# # query = "Which IP addresses made queries to usma.bluenet?"
# # retrieved_docs = vectordb.similarity_search(query, k=3)

# # print("\nRetrieved Documents:")
# # for i, doc in enumerate(retrieved_docs):
# #     print(f"Document {i + 1}: {doc.page_content}")
# #     print("-" * 50)


# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# # Load LLaMA model (e.g., LLaMA 2 or a similar model)
# model_name = "meta-llama/Llama-2-7b-chat-hf"  # Replace with your preferred LLaMA model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="cuda",  # Automatically uses GPU if available
#     torch_dtype="auto",  # Use mixed precision if supported
# )

# # Create a text generation pipeline
# llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# print(f"Model '{model_name}' loaded successfully!")


# from langchain.chains.query_constructor.base import AttributeInfo
# from langchain.retrievers.self_query.base import SelfQueryRetriever

# # Define metadata fields
# metadata_field_info = [
#     AttributeInfo(
#         name="date",
#         description="The timestamp of the log entry",
#         type="string",
#     ),
#     AttributeInfo(
#         name="client_ip",
#         description="The IP address of the client making the query",
#         type="string",
#     ),
#     AttributeInfo(
#         name="query_type",
#         description="The type of query made in the log (e.g., A, PTR, MX)",
#         type="string",
#     ),
# ]

# # Use a prompt template to guide the retriever
# from langchain.prompts import PromptTemplate
# self_query_prompt = PromptTemplate.from_template(
#     "Given the following log metadata fields: {metadata_field_info}, reformulate the user's query to retrieve the most relevant documents."
# )

# # Initialize the SelfQueryRetriever
# retriever = SelfQueryRetriever.from_llm(
#     llm=llm_pipeline,
#     vectorstore=vectordb,  # Your FAISS vector store
#     document_content_description="A DNS log entry with metadata",
#     metadata_field_info=metadata_field_info,
#     prompt_template=self_query_prompt,
# )


# from langchain.chains import RetrievalQA

# # Set up the Retrieval QA chain
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm_pipeline,
#     retriever=retriever,
#     return_source_documents=True,  # Return retrieved documents for debugging
# )

# print("RetrievalQA chain set up successfully.")


# query = "Which IP addresses made queries to usma.bluenet on November 7th, 2009?"
# response = qa_chain({"query": query})

# # Display the answer
# print("\nAnswer:")
# print(response["result"])

# # Display the source documents
# print("\nSource Documents:")
# for i, doc in enumerate(response["source_documents"]):
#     print(f"Document {i + 1}: {doc.page_content}")
#     print("-" * 50)

from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

#load embeddings and vector store 
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space":"cosine"}
)

#search for relevant documents
query = "Which is the founder of nvidia, when was it founded, and where is the headquarter located?"

retriever = db.as_retriever(search_kwargs={"k": 3})


relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
#display the results

print("___context ___")
for i,doc in enumerate(relevant_docs,1):
    print(f"Document {i}:\n {doc.page_content}\n")
    

combined_input = f"""Based on the following documents, please answer this question:{query}

Documents:
    {chr(10).join([f"-{doc.page_content}" for doc in relevant_docs])}
    
    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer the question based in the provide documents" """
  
model = ChatGoogleGenerativeAI(model="models/gemini-3.8-flash", temperature=0)

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content=combined_input),
]  

result = model.invoke(messages)

print("\n -----Generated Response----")

print("content only:")
print(result.text)
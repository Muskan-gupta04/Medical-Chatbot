import os

from langchain_groq import ChatGroq  # Correct import for Groq LLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
# Make sure you have your Groq API key set as environment variable "GROQ_API_KEY"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))
# Initialize Groq LLM
llm = ChatGroq(
    model_name="deepseek-r1-distill-llama-70b",  
    temperature=0.5,
    groq_api_key=os.environ.get("GROQ_API_KEY")
)

# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=db.as_retriever(search_kwargs={'k': 3}),
#     return_source_documents=True,
#     chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
# )


def retrieve_docs(query):
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return retriever.get_relevant_documents(query)

def get_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])

def answer_query(query):
    docs = retrieve_docs(query)
    context = get_context(docs)
    prompt = ChatPromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE)
    chain = prompt | llm
    answer = chain.invoke({"question": query, "context": context})
    return answer, docs



user_query = input("Write Query Here: ")
# response = qa_chain.invoke({'query': user_query})

# print("RESULT: ", response["result"])
# print("SOURCE DOCUMENTS: ", response["source_documents"])
user_query = input("Write Query Here: ")
answer, source_docs = answer_query(user_query)

print("RESULT:", answer)
print("\nSOURCE DOCUMENTS:")
for i, doc in enumerate(source_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n---")


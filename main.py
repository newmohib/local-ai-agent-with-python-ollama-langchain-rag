from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")
#
# template = """
# You are an exeprt in answering questions about a pizza restaurant
#
# Here are some relevant reviews: {reviews}
#
# Here is the question to answer: {question}
# """

template = """
You are an expert in answering questions about a pizza restaurant.

Use ONLY the reviews provided below.
If the answer is not contained in the reviews, say:
"I don't know from the provided reviews."

Reviews:
{reviews}

Question:
{question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break

    # reviews = retriever.invoke(question)
    # result = chain.invoke({"reviews": reviews, "question": question})
    reviews_docs = retriever.invoke(question)
    reviews_text = "\n\n".join([doc.page_content for doc in reviews_docs])

    result = chain.invoke({"reviews": reviews_text, "question": question})
    print(result)
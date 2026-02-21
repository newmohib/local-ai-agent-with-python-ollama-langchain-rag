from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about a pizza restaurant.

Write your answer in well-formatted paragraphs.
Use line breaks between ideas.

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
    question = input("Ask your question (q to quit): ").strip()
    print("\n\n")
    if question.lower() == "q":
        break

    # 1) Retrieve docs
    reviews_docs = retriever.invoke(question)

    # 2) Convert docs -> text for prompt
    reviews_text = "\n\n".join([doc.page_content for doc in reviews_docs])

    # 3) STREAM the response
    print("Answer:\n", end="", flush=True)
    for chunk in chain.stream({"reviews": reviews_text, "question": question}):
        # chunk is usually a string for OllamaLLM
        print(chunk, end="", flush=True)

    print()  # newline after streaming completesz

    # print("\nAnswer:\n")
    #
    # for chunk in chain.stream({"reviews": reviews_text, "question": question}):
    #     if chunk:
    #         print(chunk, end="", flush=True)
    #
    # print("\n\n-------------------------------")


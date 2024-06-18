from flask import Flask, request, jsonify
import os
import uuid
import re
from flask_cors import CORS
from typing import List, Tuple

from IPython.display import display, Image, Markdown

# from langchain.prompts import PromptTemplate
# from langchain.retrievers.multi_vector import MultiVectorRetriever
# from langchain.storage import InMemoryStore

# from langchain_community.vectorstores import Chroma

# from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

# from langchain_text_splitters import CharacterTextSplitter
# from flask import jsonify
from langchain_google_vertexai import (
    # VertexAI,
    ChatVertexAI,
    # VertexAIEmbeddings,
    VectorSearchVectorStore,
)
from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import CharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

loader = TextLoader('./AccountManagementSystem-AMS.txt')
docs = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(docs)



actualText = [d.page_content for d in docs]

embedding_model = VertexAIEmbeddings(model_name="textembedding-gecko@003")

vector_store = Chroma(persist_directory="./rag_content",
    collection_name="rag_data", embedding_function=embedding_model
)

retriever2 = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 1, "score_threshold": 0.4})

retriever2.vectorstore.add_documents(docs)

print(retriever2.get_relevant_documents("what is Amazon notification service?"))




system = ("You name is Service Integrator AI."
          "Your job is to help users with their queries related to software integration.\n"
          "You will be given contextual text usually system or api documentation of multiple systems.\n"
          "Use this information only to respond to the user's question. \n"
          "You are not allowed to provide any information that is not present in the given context.\n"
          "For technical integration questions, cover following information only.\n"
          "1. Pre-requisites to integrate \n"
          "2. Steps to  integrate \n"
          "3. Integration System Architecture\n"
          "4. API Documentation\n"
          "5. User Guide\n"

          # "If asked for suggestion about what user can ask, reply back with following format: \n"
          # "<Suggestion> {Suggestion 1} </Suggestion>"
          # "<Suggestion> {Suggestion 2} </Suggestion>"
          # "<Suggestion> {Suggestion 3} </Suggestion>"

          "Keep your answers grounded to contextual text only, do not hallucinate or deviate from user question. \n"
          "If provided context is not enough to answer user query then reply back with 'Sorry ; I do not have enough information to respond your queries.'")



def validate_chat_history(chat_history):
    """
    Validates that chat_history contains alternating HumanMessage and AIMessage objects.
    If the last message is a HumanMessage and the list is not alternating, removes the last message.
    """
    if not chat_history:
        return chat_history

    def is_alternating(messages):
        for i in range(len(messages) - 1):
            if isinstance(messages[i], HumanMessage) and isinstance(messages[i + 1], HumanMessage):
                return False
            if isinstance(messages[i], AIMessage) and isinstance(messages[i + 1], AIMessage):
                return False
        return True

    if is_alternating(chat_history):
        return chat_history

    if isinstance(chat_history[-1], HumanMessage):
        chat_history.pop()

    last_4_messages = chat_history[-4:]
    return last_4_messages


chat_history = []

chat_history_record = []

stable_chat_history = [
                        HumanMessage(content="Whats your name?"),
                        AIMessage(content="My name is Wells Service AI"),
                        # HumanMessage(content="can you suggest some questions user can ask?"),
                        # AIMessage(content=(["Authentication and authorization for AMS API calls?",
                        #                    "Implement real-time updates using Kafka topics?",
                        #                    "Integrate with other systems using the AMS APIs?"]))
                        ]




def img_prompt_func(data_dict):
    """
    Join the context into a single string
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = [
        {
            "type": "text",
            "text": (
                # "You are software system architect tasking with providing system architecture advice.\n"
                # "You will be given contextual text usually system documentation.\n"
                # "Use this information to provide advice and suggestions related to the user's question. \n"
                f"User-provided question: {data_dict['question']}\n\n"
                "Context:\n"
                f"{formatted_texts}"
            ),
        }
    ]

    # Adding image(s) to the messages if present
    # if data_dict["context"]["images"]:
    #     for image in data_dict["context"]["images"]:
    #         messages.append(
    #             {
    #                 "type": "image_url",
    #                 "image_url": {"url": f"data:image/jpeg;base64,{image}"},
    #             }
    #         )
    # system = "I am a Developer and I want you to return information for the given question only from the provided as Context Text"
    return [("system", system),HumanMessage(content=messages)]
    # pp = ChatPromptTemplate.from_messages([
    #     ("system", system),
    #     # prompt,
    #     [
    #         ("human", "Hello!"),
    #         AIMessage(content="Hi there"),
    #     ],
    #     HumanMessage(content=messages)
    # ])
    # print(pp)
    # return pp

def create_prompt_for_chat(data_dict):
    print(data_dict)
    """
    Join the context into a single string
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    history_messages = [SystemMessage(content=system)]
    # print(data_dict["history"])
    # for message in data_dict["history"]:
    #     if message["type"] == "human":
    #         history_messages.append(HumanMessage(content=message["text"]))
    #     elif message["type"] == "AI":
    #         history_messages.append(AIMessage(content=message["text"]))
    # print(prompt.format_messages)
    history_messages.extend(stable_chat_history)
    history_messages.extend(validate_chat_history(chat_history_record))
    # history_messages.extend(validate_chat_history(chat_history))
    print("New message history = "+str(len(history_messages)))
    context_section = f"\n\nContext:\n{formatted_texts}" if formatted_texts else ""

    messages = history_messages + [
        HumanMessage(content=(
            f"User-provided question: {data_dict['question']}\n\n"
            f"{context_section}"
        ))
    ]

#     contextualize_q_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "I am a Developer and I want you to return information for the given question only from the provided Context Text and Chat History."),
#         prompt,
#         ("human", f"User-provided question: {data_dict['question']}\n\n"
#             "Context:\n"
#             f"{formatted_texts}"),
#     ]
# )

    print(messages)
    return messages

def split_image_text_types(docs):
    print("Docus selected")
    print(len(docs))
    texts = []
    for doc in docs:
        # print(doc)
        texts.append(doc.page_content)
        # texts.append(doc.decode('utf-8'))
        # print(doc)
    return {"texts": texts}

def extract_query(data_dict):
    print(data_dict["question"])
    return data_dict["question"]

def extract_history(data_dict):
    print(data_dict)
    return data_dict["history"]

# Create RAG chain
chain_multimodal_rag = (
    {
        "context": retriever2 | RunnableLambda(split_image_text_types),
        "question": RunnablePassthrough(),
        # "history": RunnablePassthrough()
    }
    | RunnableLambda(create_prompt_for_chat)
    | ChatVertexAI(
        temperature=0, model_name="gemini-pro", max_output_tokens=1024, top_k=10
    )
)

# Suggestion set : START

class Suggestion(BaseModel):
    suggestion: str = Field(description="suggested question to ask")
    suggestion_type: str = Field(description="type of suggestion either 'question' or 'instruction")
parser = JsonOutputParser(pydantic_object=Suggestion)
parser.get_format_instructions()


def suggestion_prompt_creator(data_dict):
    print(data_dict)
    """
    Join the context into a single string
    """
    # formatted_texts = "\n".join(data_dict["context"]["texts"])
    history_messages = [SystemMessage(content="You are AI support agent, who provide suggestions what next user may ask based on chat history. Do not include any suggestions that require internet and do not assume any information.")]
    history_messages.extend(validate_chat_history(chat_history_record))

    messages = history_messages + [
        HumanMessage(content=(data_dict['question']
        ))
    ]
    return messages

suggestion_chain = (
    {
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(suggestion_prompt_creator)
    | ChatVertexAI(
        temperature=0, model_name="gemini-pro", max_output_tokens=1024, top_k=10
    )
)

# Suggestion set : END


# question = "What is AMS system?"
# result = chain_multimodal_rag.invoke(question)
# if(result.content != ""):
#     chat_history_record.extend([HumanMessage(content=question), AIMessage(content=result.content)])
# # result
# Markdown(result.content)


app = Flask(__name__)
CORS(app)
@app.route('/test')
def test():
    return 'Hello server up'

@app.route('/hello')
def hello():
    print("Hello API called")
    question = "What is AMS system?"
    result = chain_multimodal_rag.invoke(question)
    return jsonify({'message': result.content})

@app.route('/submit', methods=['POST'])  
def submit_data():
    data = request.json  # Assuming the client sends JSON data
    # Process the data (e.g., store in database, perform calculations)
    # Example: storing data in a variable
    # received_data = {
    #     'prompt': data['prompt']
    # }

    question = data['prompt']
    result = chain_multimodal_rag.invoke(question)
    suggestions = []
    if(result.content != ""):
        chat_history_record.extend([HumanMessage(content=question), AIMessage(content=result.content)])
        if(len(chat_history_record)>3):
            query = ("can you suggest 4 brief and direct instructions/question with short interrogative phrases that user can give to you based on your last response?")
            query = query + parser.get_format_instructions()
            suggestions_res = suggestion_chain.invoke(query)
            suggestions = parser.parse(suggestions_res.content)
        # print('SUggestions')
        # print(suggestions.content)
        # Markdown(suggestions.content)

    # result
    # Markdown(result.content)
    return jsonify({'message': 'success', 'message': result.content, 'suggestions':suggestions})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

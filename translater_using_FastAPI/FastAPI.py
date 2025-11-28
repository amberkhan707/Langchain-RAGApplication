#Enviroment
import os 
from dotenv import load_dotenv
load_dotenv()
os.environ["groq_api_key"] = os.getenv("groq_api_key")

#Creation of Model
from langchain_groq import ChatGroq
model = ChatGroq(model= "qwen/qwen3-32b")

#Writing Prompt
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a highly skilled professional translator. "
            "Translate the text accurately. "
            "Never include chain-of-thought, hidden reasoning, intermediate steps, or <think> tags. "
            "Only return the final translated text. "
            "No explanations. No analysis."
        ),
        (
            "user",
            "Translate the following text into {target_language}:\n\n{text}"
        )
    ]
)


#Writing Output Parser
from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()

chain = prompt | model | parser

#Creating APP using FastAPI
from langserve import add_routes
from fastapi import FastAPI

app = FastAPI()

add_routes(
    app,
    chain,
    path = "/chain"
)

if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app, host = "localhost", port = 8080)
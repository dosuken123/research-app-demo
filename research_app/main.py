from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.schema import StrOutputParser, Document
from langchain.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from web_loader import scrape_text
from web_search import web_search

load_dotenv(".env")

template = """
Context: {context}

--------
Answer the following question based on the context above.

Question: {question}
--------
"""

prompt = ChatPromptTemplate.from_template(template)

sub_chain = (
    RunnablePassthrough.assign(
        context=lambda x: scrape_text(x['url'])[:5000]
    ) |
    prompt |
    ChatOpenAI(model="gpt-3.5-turbo-1106") |
    StrOutputParser()
)

chain = (
    RunnablePassthrough.assign(
        urls=lambda x: web_search(x['question'], num_of_results=3)
    ) |
    RunnableLambda(lambda x: [{"url": url, "question": x['question']} for url in x['urls']]) |
    sub_chain.map()
)

chain.invoke(
    {
        "question": "What kind of character Luke is in Street Fighter 6?"
    }
)

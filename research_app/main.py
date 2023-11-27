from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.schema import StrOutputParser, Document
from langchain.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from web_loader import scrape_text
from web_search import web_search
import json

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

webscraping_chain = (
    RunnablePassthrough.assign(
        urls=lambda x: web_search(x['question'], num_of_results=3)
    ) |
    RunnableLambda(lambda x: [{"url": url, "question": x['question']} for url in x['urls']]) |
    sub_chain.map()
)


question_generation_prompt = ChatPromptTemplate.from_messages(
    ("user",
    "Write 3 google search queries to search online that form an "
    "objective opinion from the following: {question}\n"
    "You must respond with a list of strings in the following format: "
    '["query 1", "query 2", "query 3"].')
)

questions_generate_chain = question_generation_prompt | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser() | json.loads

research_chain = questions_generate_chain | (lambda x: [{"question": q} for q in x]) | webscraping_chain.map()


WRITER_SYSTEM_PROMPT = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."  # noqa: E501


# Report prompts from https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/master/prompts.py
RESEARCH_REPORT_TEMPLATE = """Information:
--------
{research_summary}
--------
Using the above information, answer the following question or topic: "{question}" in a detailed report -- \
The report should focus on the answer to the question, should be well structured, informative, \
in depth, with facts and numbers if available and a minimum of 1,200 words.
You should strive to write the report as long as you can using all relevant and necessary information provided.
You must write the report with markdown syntax.
You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
Write all used source urls at the end of the report, and make sure to not add duplicated sources, but only one reference for each.
You must write the report in apa format.
Please do your best, this is very important to my career."""  # noqa: E501


summary_prompt = ChatPromptTemplate.from_messages(
    (
        ("system", WRITER_SYSTEM_PROMPT),
        ("user", RESEARCH_REPORT_TEMPLATE)
    )
)

def join_list_in_list(list_in_list):
    content = []
    for l in list_in_list:
        content.append("\n\n".join(l))
    return "\n\n".join(content)

chain = RunnablePassthrough.assign(
    research_summary= research_chain | join_list_in_list,
) | summary_prompt | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()

# chain.invoke(
#     {
#         "question": "What kind of character Ryu is in Street Fighter 6?"
#     }
# )

#!/usr/bin/env python
from fastapi import FastAPI
from langserve import add_routes


app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    chain,
    path="/research-app",
)

# add_routes(
#     app,
#     ChatAnthropic(),
#     path="/anthropic",
# )

# model = ChatAnthropic()
# prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
# add_routes(
#     app,
#     prompt | model,
#     path="/joke",
# )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)

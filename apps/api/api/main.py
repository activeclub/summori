import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import html2text
import logfire
import requests
from bs4 import BeautifulSoup
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from api.config import config
from api.discord import send_message

logfire.configure()


def query_pydantic_ai(url: str):
    r = requests.get(url)
    text = html2text.html2text(r.text)
    prompt = f"""Please prepare a summary of the paper you have given in the following format in Japanese.
Each item should be expressed in three bullet points.

【背景】
【目的】
【手法】
【実験方法】
【実験結果】
【考察】

The paper begins here:
---
{text}"""

    model = OpenAIModel(model_name="o1-preview", api_key=config.openai_api_key)
    agent: Agent = Agent(model=model)

    result = agent.run_sync(prompt)
    return result.data


def query_langchain(url: str):
    h2t = Html2TextTransformer()

    loader = AsyncHtmlLoader([url])
    docs = loader.load()
    assert len(docs) == 1

    bs = BeautifulSoup(docs[0].page_content, features="html.parser")
    img_tags = bs.find_all("img")

    images = []
    for img_tag in img_tags:
        if img_tag["src"].startswith("data:"):
            images.append(img_tag["src"])
        else:
            images.append(f"{url}/{img_tag['src']}")

    docs_transformed = h2t.transform_documents(docs)

    prompt = ChatPromptTemplate.from_template(
        """Please prepare a summary of the paper you have given in the following format in Japanese.
Each item should be expressed in three bullet points.

【背景】
【目的】
【手法】
【実験方法】
【実験結果】
【考察】

The paper begins here:
---
{context}"""
    )

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=config.openai_api_key)  # type: ignore
    chain = create_stuff_documents_chain(llm, prompt)
    result = chain.invoke({"context": docs_transformed})

    return result


def load_rss(url: str) -> dict:
    r = requests.get(url)
    root = ET.fromstring(r.text)

    channel = root.find("channel")
    if channel is None:
        return {}

    rss_dict: dict = {"title": None, "items": []}
    rss_dict["title"] = getattr(channel.find("title"), "text", "")

    items = channel.findall("item")
    for item in items:
        item_dict: dict = {}
        item_dict["title"] = getattr(item.find("title"), "text", "")
        item_dict["link"] = getattr(item.find("link"), "text", "")
        item_dict["description"] = getattr(item.find("description"), "text", "")
        published_at_str = getattr(item.find("pubDate"), "text", "")
        item_dict["published_at"] = datetime.strptime(
            published_at_str, "%a, %d %b %Y %H:%M:%S %z"
        )
        rss_dict["items"].append(item_dict)

    return rss_dict


if __name__ == "__main__":
    rss_dict = load_rss(url="https://rss.arxiv.org/rss/cs.RO")
    now = datetime.now()
    for page in rss_dict["items"][:5]:
        if now - page["published_at"] <= timedelta(minutes=60):
            summarized_text = query_pydantic_ai(page["link"])
            send_message(
                message=f"""{page["title"]}
        {page["link"]}

        {summarized_text}
        """
            )

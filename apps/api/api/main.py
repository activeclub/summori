import xml.etree.ElementTree as ET
from pprint import pprint

import requests
from bs4 import BeautifulSoup
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import AsyncHtmlLoader, WebBaseLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from api.config import config

llm = ChatOpenAI(model="gpt-4o-mini", api_key=config.openai_api_key)  # type: ignore


def load_web(url: str):
    loader = WebBaseLoader(web_paths=[url])

    docs = loader.load()

    doc = docs[0]
    print(doc)


def load_html2text(url: str):
    html2text = Html2TextTransformer()

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

    docs_transformed = html2text.transform_documents(docs)

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
        rss_dict["items"].append(item_dict)

    return rss_dict


if __name__ == "__main__":
    rss_dict = load_rss(url="https://rss.arxiv.org/rss/cs.RO")
    page = rss_dict["items"][0]
    summarized_text = load_html2text(page["link"])

    print(page["title"])
    print("-----")
    pprint(summarized_text)

from pprint import pprint

from bs4 import BeautifulSoup
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import AsyncHtmlLoader, WebBaseLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from api.config import config

llm = ChatOpenAI(model="gpt-4o-mini", api_key=config.openai_api_key)

page_url = "https://arxiv.org/html/2412.05313v1"


def load_web():
    loader = WebBaseLoader(web_paths=[page_url])

    docs = loader.load()

    doc = docs[0]
    print(doc)


def load_html2text():
    html2text = Html2TextTransformer()

    loader = AsyncHtmlLoader([page_url])
    docs = loader.load()
    assert len(docs) == 1

    bs = BeautifulSoup(docs[0].page_content, features="html.parser")
    img_tags = bs.find_all("img")

    images = []
    for img_tag in img_tags:
        if img_tag["src"].startswith("data:"):
            images.append(img_tag["src"])
        else:
            images.append(f"{page_url}/{img_tag['src']}")

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

    pprint(result)


if __name__ == "__main__":
    load_html2text()

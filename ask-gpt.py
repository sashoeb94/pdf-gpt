"""Program to convert given PDF to text and run queries on it based on GPT"""
import os
import re
import sys

import fitz
import openai
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY


def preprocess(text):
    text = text.replace('\n', ' ')
    text = text.replace('•', '')
    text = re.sub('\s+', ' ', text)
    return text

def pdf_to_text(path, start_page=1, end_page=None):
    """Converts PDF to text"""
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []

    for i in range(start_page - 1, end_page):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list

def write_to_file(text):
    with open('data.txt', 'w') as f:
        for line in text:
            f.write(line)
    f.close()

def main(query, filePath, start_page, end_page, persist=False):
    text = pdf_to_text(filePath, start_page, end_page)
    write_to_file(text)
    print("Done converting PDF to text data.")

    if persist and os.path.exists("persist"):
        print("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        from langchain.indexes.vectorstore import VectorStoreIndexWrapper
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        loader = TextLoader('data.txt')
        if persist:
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator().from_loaders([loader])

    print("Running query...")
    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )
    print(chain.run(query))

if __name__ == "__main__":
    query = sys.argv[1]
    filePath = sys.argv[2]
    start_page = int(sys.argv[3])
    end_page = int(sys.argv[4])
    # Enable to cache & reuse the model to disk (for repeated queries on the same data)
    persist = False
    print(f"Converting {filePath} to text pages {start_page}-{end_page} and running query: {query}")
    main(query, filePath, start_page, end_page, persist)

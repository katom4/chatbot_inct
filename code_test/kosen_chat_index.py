from langchain.document_loaders import BSHTMLLoader
from langchain.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
import langchain

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

import sys

langchain.verbose = True


from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams

if False:
    page_lists = [[1],[2],[4],[6],[7],[8,9],[10],[11,12,13],[14],[15,16],[19],[21],[23],[25],[27],[29],[31,32],[41],[42],[43],[44],[45],[46],[47],[56]]
    texts = []

    laparams = LAParams()

    for page_list in page_lists:
        text = extract_text(
            pdf_file="document/ishikawa_yoran.pdf",
            #page_numbers = page_list
            )
        texts.append(text)

    print(text)

    sys.exit()



    text_splitter = RecursiveCharacterTextSplitter(
        #separators = ["."] , # セパレータ
        #日本語
        separators = ["。"],
        #separator = "。",
        chunk_size = 200,  # チャンクの文字数
        chunk_overlap = 0,  # チャンクオーバーラップの文字数
    )

    """
    text_splitter = CharacterTextSplitter(
        #separators = ["."] , # セパレータ
        #日本語
        #separators = ["。"],
        separator = "。",
        chunk_size = 100,  # チャンクの文字数
        chunk_overlap = 0,  # チャンクオーバーラップの文字数
    )
    """
    texts = []
    with open('./document/ishikawa_yoran.txt', 'r') as file:
        data = file.read()
        texts = text_splitter.split_text(data)



    print(len(texts))

    import re
    # 正規表現で、英語（もしくは空白？）が10〜15文字程度続いた場合は、テキストデータから削除する処理を挟む

    replace_texts = []
    for text in texts:
        if(text[0] == "。"):
            text = text[1:]
        if(text[-1] == "。"):
            text = text[:-1]
        rt = text.replace('\n','').replace(' ','').replace('\u3000','')
        #m = re.search(r'([a-zA-Z]|\s){30,}', rt)
        rt2 = re.sub(r'([a-zA-Z]|\s|\.|,|\d|\(|\)|\'|\"|’|”){15,}','' ,rt)
        #if len(rt) < 500:
        replace_texts.append(rt2)

    processed_texts = list(dict.fromkeys(replace_texts))
    """
    for i,text in enumerate(replace_texts):
        if i == 0:
            continue
        if text != replace_texts[i-1]:
            processed_texts.append(text)
    """

    print(len(processed_texts))

    import csv

    with open('./document/processed_text_re.txt', 'w') as file:
        for text in processed_texts:
            file.write(text+"\n")

    '''

    '''


    #urlからデータの読み取り（ページによってはできてない場合があるかも？要テスト）

    urls = [
    "https://www.ishikawa-nct.ac.jp/adm/policy.html"
        ]


    #loader = UnstructuredURLLoader(urls=urls)

    #data = loader.load()

    loader = PyPDFLoader("document/kosen_guide_4p.pdf")

    pages = loader.load_and_split()

    print(pages[0])

    text_splitter = RecursiveCharacterTextSplitter(
        #separators = ["."] , # セパレータ
        #日本語
        separators = ["。"],
        chunk_size = 100,  # チャンクの文字数
        chunk_overlap = 0,  # チャンクオーバーラップの文字数
    )

    texts = []
    for d in data:
        texts += text_splitter.split_text(d.page_content)
        texts += "。"


    #replace_texts = [text.replace('.', '').replace('\n', '') for text in texts]

    #日本語
    replace_texts = [text.replace('。', '').replace('\n', '') for text in texts]

    print(replace_texts)

    print(len(replace_texts))
    for text in replace_texts:
        print(text[:10].replace("\n", "\\n"), ":", len(text))




with open('./document/processed_text_self_title.txt') as f:
    l_strip = [s.rstrip() for s in f.readlines()]
    prefix_list = ["passage: " + s for s in l_strip]

# インデックスの作成


index = FAISS.from_texts(
    #texts=processed_texts,
    texts = prefix_list,
    embedding=HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large"),
)
index.save_local("storage_kosen")

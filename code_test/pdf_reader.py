from langchain.document_loaders import UnstructuredFileLoader

loader = UnstructuredFileLoader("/root/LLM/kosen_chat/document/arg_2.pdf",mode = "elements", strategy="hi_res")
docs = loader.load()
print(f"number of docs: {len(docs)}")
print(docs[0].page_content)
print("\n----------\n".join([str(e.page_content) for e in docs]))


from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def get_answers(db,question, num = 4):
    query = "query: " + question
    docs = db.similarity_search(
        query = query,
        k = num )
    return docs

    return answers

#返されるスコアはL2ノルム（スコアが低い方が類似度が高い）
def get_answers_with_score(db, question, num = 4):
    query = "query: " + question
    docs = db.similarity_search_with_score(
        query = query,
        k = num )

    return docs

if __name__ == "__main__":
    db = FAISS.load_local(
    "./storage_kosen", HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    )
    question = "環境都市工学科と建築学科の違いは何ですか？"
    answers_with_score = get_answers_with_score(db,question,num = 15)
    file_pass = "./emb_test.txt"
    with open(file_pass, mode = "w") as f:
        for answer in answers_with_score:
            print("text : " , answer[0].page_content , "score : " , answer[1])
            f.write("text : " + answer[0].page_content + "score : " + answer[1] + "\n")

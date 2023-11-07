from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.prompts import FewShotPromptTemplate,PromptTemplate
from langchain.chains import LLMChain,SequentialChain


from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

from auto_gptq import exllama_set_max_input_length

import torch
import sys

import langchain

from prompt_xwin import template_fewshot, template, B_INST, E_INST, B_SYS, E_SYS, DEFAULT_SYSTEM_PROMPT, few_shot_context, few_shot_question, few_shot_answer

#langchain.verbose = True
langchain.debug = True




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
    prompt = PromptTemplate(
            input_variables=["question","context"],
            template = template_fewshot,
        )
    model_name = "TheBloke/Xwin-LM-70B-V0.1-GPTQ"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast = True)
    
    eos_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        #use_flash_attention_2=True,
        device_map="auto"
        )
    model = exllama_set_max_input_length(model, 4096)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature = 0.01
        #device_map=1
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    db = FAISS.load_local(
    "./storage_kosen", HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    )
    

    chain = LLMChain(
        llm = llm,
        prompt = prompt,
    )
    
    print("--------------------")
    print("石川高専bot : 石川高専に関する質問に答えます。「exit」と入力すると終了します。\n")
    while True:
        question = input("ユーザー : ")
        print("")

        if question == "exit":
            print("石川高専bot : チャットを終了します。ありがとうございました。")
            sys.exit()
        

        answers = get_answers(db,question,num = 4)
        answers_with_score = get_answers_with_score(db,question,num = 4)

        #print("text : " , answers[0].page_content, "metadata : ", answers[0].metadata)
        for ans in answers_with_score:
            print("text : " , ans[0].page_content , "score : " , ans[1] , "metadata : " , ans[0].metadata)

        #質問と返答のスコア（一致度）が悪い場合
        if answers_with_score[0][1] > 0.3:
            print("石川高専bot : 私の能力ではこの質問に回答することができません。申し訳ございません。\n")
            continue

        context = "\n\n"
        for ans in answers:
            context += ans.page_content[9:] #prefixの除去
            context += "\n\n"


        result = chain.run({
            "question" : question,
            "context" : context
        })

        print("石川高専bot : " + result + "\n")
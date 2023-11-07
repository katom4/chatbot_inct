from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.prompts import FewShotPromptTemplate,PromptTemplate
from langchain.chains import LLMChain,SequentialChain


from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
import torch
import sys

import langchain

from prompt_llama import prompt_template_fewshot, prompt_template, B_INST, E_INST, B_SYS, E_SYS, DEFAULT_SYSTEM_PROMPT, few_shot_context, few_shot_question, few_shot_answer

langchain.verbose = True




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
    model_name = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
    #model_name = "elyza/ELYZA-japanese-Llama-2-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    eos_token = tokenizer.eos_token

    prompt_fewshot_2 = "{bos_token}{b_inst} {system}{context}{question}{e_inst}{answer}{eos_token}".format(
        bos_token=tokenizer.bos_token,
        b_inst=B_INST,
        system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
        context = few_shot_context,
        question = few_shot_question,
        e_inst = E_INST,
        answer = few_shot_answer,
        eos_token = tokenizer.eos_token
    ) + "{bos_token}{b_inst} ".format(
        bos_token=tokenizer.bos_token,
        b_inst=B_INST,
    )+ "{context}{question}" + "{e_inst}".format(e_inst=E_INST)



    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        #device_map="auto"
        device_map = 1
        )

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
    prompt = PromptTemplate(
            input_variables=["question","context"],
            template = prompt_fewshot_2,
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
        
        question = "質問:" + question
        
        answers = get_answers(db,question,num = 4)
        answers_with_score = get_answers_with_score(db,question,num = 4)

        #print("text : " , answers[0].page_content, "metadata : ", answers[0].metadata)
        for ans in answers_with_score:
            print("text : " , ans[0].page_content , "score : " , ans[1] , "metadata : " , ans[0].metadata)

        #質問と返答のスコア（一致度）が悪い場合
        if answers_with_score[0][1] > 0.3:
            print("石川高専bot : 私の能力ではこの質問に回答することができません。申し訳ございません。\n")
            continue

        context = "情報を以下に示します\n\n"
        for ans in answers:
            context += ans.page_content[9:] #prefixの除去
            context += "\n\n"


        result = chain.run({
            "question" : question,
            "context" : context
        })

        print("石川高専bot : " + result + "\n")
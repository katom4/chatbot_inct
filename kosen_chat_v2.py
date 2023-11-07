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
import time

#from prompt_xwin import prompt_template_fewshot, prompt_template, B_INST, E_INST, B_SYS, E_SYS, DEFAULT_SYSTEM_PROMPT, few_shot_context, few_shot_question, few_shot_answer


langchain.verbose = True

system_ret = """あなたは与えられた仕事を正確に行う日本語処理システムです。
今から石川高専に関する質問文を提示するので、それぞれの質問に回答するのに「どのような情報が必要か」を考えて出力してください。
従うべきいくつかのルール:
1. 箇条書きで「質問に回答するにはどのような情報が必要か」という内容のみを出力してください。その内容は、具体的な細かい内容にしてください。
2. 出力する内容は最小で1個、最大で4個にする必要があります。
3. 「質問に回答するにはどのような情報が必要か」という内容は、可能な限りわかりやすい文章にする必要があります。


USER: 電子情報工学科と電気工学科の違いは何ですか？
ASSISTANT: ・電子情報工学科の概要 ・電気工学科の概要 ・電子情報工学科と電気工学科のカリキュラムの違い

USER: 建築学科で学べることを教えてください。
ASSISTANT: ・建築学科の概要 ・建築学科のカリキュラム ・建築工学科の科目

USER: {question}
ASSISTANT: """

system_ans = """あなたは世界中で信頼されている質問回答システムです。
事前知識ではなく、常に提供された質問に関連する情報を用いて質問に回答してください。
従うべきいくつかのルール:
1. 与えられた情報の中には、質問の回答に関係のない情報が入っている場合があります。その場合は、該当する情報を無視してください。与えられた情報を全て用いても、質問に回答することができないと判断した場合は、質問に回答せず「情報なし」と出力してください。
2. 回答内で提示された情報を直接参照しないでください。
3. 「...のような情報が与えられています。」や「与えられた情報によると..」、またはそれに類いするような記述は避けてください。

情報は以下の通りです。
{{
{context}
}}
事前知識ではなく提供された情報を考慮して質問に答えてください。

USER: {question}
ASSISTANT: """

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
    prompt_retrieval = PromptTemplate(
            input_variables=["question"],
            template = system_ret,
        )
    
    prompt_answer = PromptTemplate(
            input_variables=["question","context"],
            template = system_ans,
        )

    #model_name = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
    model_name = "TheBloke/Xwin-LM-70B-V0.1-GPTQ"
    #model_name = "elyza/ELYZA-japanese-Llama-2-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        #device_map="auto"
        device_map = "auto",
        )

    model = exllama_set_max_input_length(model, 4096)

    pipe_ret = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        do_sample = True,
        temperature = 0.001,
        top_p=0.95,
        #device_map=1
    )

    pipe_ans = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        do_sample = True,
        temperature = 0.001,
        top_p=0.95,
        #device_map=1
    )

    llm_ret = HuggingFacePipeline(pipeline=pipe_ret)
    llm_ans = HuggingFacePipeline(pipeline=pipe_ans)

    db = FAISS.load_local(
    "./storage_kosen", HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    )

    question = "建築工学科と環境都市工学科の違いを教えてください。"
    chain_retrieval = LLMChain(
        llm = llm_ret,
        prompt = prompt_retrieval,
    )

    chain_answer = LLMChain(
        llm = llm_ans,
        prompt = prompt_answer,
    )
    
    print("--------------------")
    print("石川高専bot : 石川高専に関する質問に答えます。「exit」と入力すると終了します。\n")
    while True:
        question = input("ユーザー : ")
        print("")
        if question == "exit":
            print("石川高専bot : チャットを終了します。ありがとうございました。")
            sys.exit()
        
        start_time = time.time()
        answer_retrieval = chain_retrieval.run({
            "question" : question

        })
        print("answer_retriebal: ",answer_retrieval)
        make_query_time = time.time() - start_time
        print("make_query_time: ", make_query_time)
        # 変なことを言い出した時にノイズにならないように、改行した後の文章は無視する
        answers_retrieval_0 = answer_retrieval.split("\n\n")
        print(answers_retrieval_0)
        answers_retrieval = answers_retrieval_0[0].split("・")
        print(answers_retrieval)

        
        contexts = []
        contexts_score = []
        context_texts = []
        n = 3
        while True:
            for ans in answers_retrieval[1:]:
                context = get_answers(db,ans,num = n)
                context_score = get_answers_with_score(db,ans,num = n)
                contexts.extend(context)
                contexts_score.extend(context_score)
            
            context_texts = [c.page_content for c in contexts]
            list(dict.fromkeys(context_texts))
            text_all = ""
            for t in context_texts:
                text_all += t
            if len(text_all) > 1000 and n > 1:
                n -= 1
                contexts = []
                contexts_score = []
                context_texts = []
            else:
                break
        
        print("----query----")
        for con in contexts:
            print("con:",con.page_content)
        print("--------")
        
        

        #print("text : " , answers[0].page_content, "metadata : ", answers[0].metadata)
        #for ans in answers_with_score:
        #    print("text : " , ans[0].page_content , "score : " , ans[1] , "metadata : " , ans[0].metadata)

        #質問と返答のスコア（一致度）が悪い場合
        #if answers_with_score[0][1] > 0.3:
        #    print("石川高専bot : 私の能力ではこの質問に回答することができません。申し訳ございません。\n")
        #    continue

        context_text = ""
        for con in contexts:
            if con.page_content[9:] not in context_text:
                context_text += con.page_content[9:] #prefixの除去
                context_text += "\n\n"
        context_text = context_text[:-2]


        result = chain_answer.run({
            "question" : question,
            "context" : context_text
        })
        total_time = time.time() - start_time
        answer_time = total_time - make_query_time
        print("answer_time :", answer_time, "total_time: ",total_time)
        print("石川高専bot : " + result + "\n")
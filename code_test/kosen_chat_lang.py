#from langchain.document_loaders import BSHTMLLoader
#from langchain.document_loaders import UnstructuredURLLoader
#from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
import langchain

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import pipeline
from langchain.llms import HuggingFacePipeline


langchain.verbose = True



#インデックスの読み込み
index = FAISS.load_local(
   "storage_kosen", HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
)

model_name = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
#model_name = "elyza/ELYZA-japanese-Llama-2-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,
    device_map=1
    )

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    device_map=1
)


llm = HuggingFacePipeline(pipeline=pipe)

#llm = LlamaCpp(
#        model_path="./models/llama-2-13b-chat.ggmlv3.q4_K_M.bin",
#        n_gpu_layers=100,
#        temperature = 0,
#        verbose = True,
#        n_ctx=2048
#        )

from langchain.prompts import PromptTemplate

#langchainデフォルトのプロンプト
'''
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer in Italian:"""
'''

#langchainデフォルトのプロンプトの和訳
'''
prompt_template = """
以下の参考用のテキストの一部を参照して、質問に日本語で回答してください。もし参考用のテキストの中に回答に役立つ情報が含まれていなければ、分からない、と答えてください。

{context}

質問: {question}
回答:
"""
'''


#ELYZA-japanese-llama2（llama2）の定型プロンプト
prompt_template_fewshot = """<s>[INST] <<SYS>>
あなたは世界中で信頼されている質問回答システムです。
情報源を提示するので、事前知識ではなく情報源からの情報を考慮して質問に答えてください。
情報源の中には質問に対して関係のない物が入っている場合があります。その場合は該当する情報源を無視してください。
情報源から質問に対する答えを得られない場合は「情報無し」と答えてください。
従うべきいくつかのルール:
1. 回答内で指定された情報源を直接参照しないでください。
2. 「情報源に基づいて、...」や「情報源は...」、「...についての情報を提供します」など、質問の回答以外の記述は避けてください。

情報源を以下に示します。

passage: 石川高専の推薦による選抜　推薦書，調査書および適性検査（数学）・面接の結果を総合して判定します。

passage: 石川高専の学力検査による選抜　学力検査の成績および調査書をもとに総合的に判定します。学力検査は，理科，英語，数学，国語の４教科について筆記試験（マークシート方式）を行い，各教科 100 点満点の合計 400 点満点とします。１教科でも受検しないと失格（不合格）になります。

passage: 石川高専の全学科共通の求める学生像　1）ものづくりに関心があり，様々な課題に意欲を持って取り組む人　2）社会のルールを守り，向上心を持って学校生活を送る人　3）将来，技術者として社会の発展に貢献したい人

passage: 石川高専の教育目標（養成すべき人材像）　1）幅広い視野を持ち，国際社会や地球環境を理解できる技術者　2）社会的責任感と技術者としての倫理観を備えた技術者　3）問題や課題を完遂するための気概と指導力，協調性を備えた技術者　4）好奇心や目的意識・職業意識が旺盛で，十分な意欲を持つ技術者　5）確実な基礎学力と体験や実技を通して備えた実践力を持つ技術者　6）自ら問題を解決する能力（事象の理解，問題の発見，課題の設定・解決）を持つ技術者　7）学習や研究の成果を論理的に記述し，発表し，討議する能力を持つ技術者　8）学んだ知識を柔軟に活用できる応用力を持つ技術者　9）地域との交流を通して積極的な社会参加の意識を持つ技術者　10）相互理解の上に立ったコミュニケーション能力を持つ技術者

<</SYS>>
query: 石川高専の選抜について教えてください
[\INST]石川高専では、推薦による選抜と学力検査による選抜の2つの方法があります。

推薦による選抜では、推薦書、調査書、適性検査（数学）、および面接の結果を総合的に評価して選抜が行われます。

学力検査による選抜では、理科、英語、数学、国語の4つの教科について筆記試験（マークシート方式）が実施され、各教科ごとに最大100点ずつの合計400点で評価されます。1つの教科でも受験しない場合は失格（不合格）となります。</s><s>[INST]<<SYS>>

情報源を以下に示します。

{context}

<</SYS>>
{question}
[/INST]"""

prompt_template = """<s>[INST] <<SYS>>
あなたは世界中で信頼されている質問回答システムです。
情報源を提示するので、事前知識ではなく情報源からの情報を考慮して、可能な限り具体的に質問に答えてください。
情報源から質問に対する答えを得られない場合は「情報無し」と答えてください。
従うべきいくつかのルール:
1. 回答内で指定された情報源を直接参照しないでください
2. 「情報源に基づいて、...」や「情報源は...」、またはそれに類するような記述は避けてください。


情報源を以下に示します。

{context}

<</SYS>>
{question}
[/INST]"""



prompt_template_plain = """
あなたは世界中で信頼されている質問回答システムです。
情報源を提示するので、事前知識ではなく情報源からの情報を考慮して、可能な限り具体的に質問に答えてください。
情報源から質問に対する答えを得られない場合は「情報無し」と答えてください。
従うべきいくつかのルール:
1. 回答内で指定された情報源を直接参照しないでください
2. 「情報源に基づいて、...」や「情報源は...」、またはそれに類するような記述は避けてください。

情報源を以下に示します。

電子情報工学科では，電子工学，情報工学，通信工学の豊富な知識をもちながら，これらを融合した技術を駆使しシステム思考のできる人材を育成すると同時に，人や環境も視野に入れた未来志向の電子情報工学技術者を育てることを目指しています。

電子情報工学科　・コンピュータの原理やプログラミングなどに興味がある人　・情報・電子・通信の融合技術を身につけたい人　・最先端の情報通信技術で社会に貢献したい人

＜電子情報工学科＞　電子情報工学科の学習目標を達成するために下記のとおり教育課程を編成しています。情報・電子・通信などの基礎知識と技術を習得するために，1 年生から 5 年生までに多くの専門科目を配置しています。実験や演習，卒業研究を通して，システム設計や開発を行うことができる能力を身につけられるようにしています。

学科紹介　電子情報工学科　スマートフォン，インターネット，SNS 等これら情報通信ツールは今や爆発的普及を見せています。その背景には，電子・情報・通信分野の高度技術が隠されており，今も絶え間なく進歩しています。技術の進歩に伴って，地球規模で様々な問題が起こっていることも事実ですが，技術の進歩を止めるわけにはいきません。人類には人間と地球の両方に利益をもたらす高度技術の開発が求められており，電子情報工学はその重要な鍵を握っています。本学科では，このような社会状況をふまえ，電子工学，情報工学，通信工学の豊富な知識を持ちながら，21 世紀の高度技術社会にふさわしいセンスを身につけた電子情報工学技術者の育成を目指しています。

質問:電子情報工学科ではどのようなことを学びますか？

回答:電子情報工学科では、情報・電子・通信の基礎知識と技術を習得するために、1年生から5年生までに多くの専門科目を提供しています。また、実験や演習、卒業研究を通して、システム設計や開発を行う能力を身につけることができます。学科の学習目標は、高度な技術社会に適したスキルとセンスを持つ電子情報工学技術者を育成することであり、これに基づいて教育課程が編成されています。</s><s>

情報源を以下に示します。

{context}

質問:{question}

回答:"""



PROMPT = PromptTemplate(
    template=prompt_template_fewshot, input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type="stuff",
    #プロンプトを変える場合
    chain_type_kwargs={"prompt":PROMPT},
    retriever=index.as_retriever(search_kwargs={"k": 4}),
    verbose=True,
)

# 質問応答チェーンの実行
result = qa_chain.run("query: 石川高専の電子情報工学科と電気工学科の違いについて教えてください")
print("----result----\n"+result)

import os
import streamlit as st
import pinecone
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Pinecone
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import threading  # 导入 threading 模块
import logging    # 导入 logging 模块
# 插入自定义CSS来修改背景颜色（#6495ED 透明度 69%）
background_css = """
<style>
    .stApp {
        background-color: rgba(100, 149, 237, 0.2);  /* Cornflower Blue with 69% transparency */
    }
</style>
"""
st.markdown(background_css, unsafe_allow_html=True)
# 加载环境变量
load_dotenv()
# OPENAI_API_KEY = "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"
OPENAI_API_KEY = "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"
PINECONE_API_KEY = '4f4a45d8-d09e-4e12-b1b2-0ab0fba4a851'
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('us-east1-gcp')

# 初始化 Pinecone 和 OpenAI
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
# os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
# pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
pinecone_instance = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment="us-east1-gcp")
# Pinecone 的索引名称
index_name = "ce322module2"

# 初始化嵌入模型和向量存储
# embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
# vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
vectorstore = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
    # pinecone_index = pinecone_instance.Index(index_name) # Remove this line
)
# 创建 GPT 模型
# llm = ChatOpenAI(model="ft:gpt-4o-2024-08-06:personal::AFhUP34L", temperature=0)
model_gpt = 'ft:gpt-4o-2024-08-06:personal::AFhUP34L'



llm = ChatOpenAI(
    model=model_gpt,
    temperature=0
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10}) #这句话原来没有
# 创建自定义提示模板
# prompt_template = PromptTemplate(
#     input_variables=["history", "question", "context"],
#     template="""You are an expert in civil engineering, and this class is about PROJECT CONTROL & LIFE CYCLE EXECUTION OF CONSTRUCTED FACILITIES.
#     {history} question: {question} context: {context} If the context provides an answer, answer based on the context. If the context is not sufficient to answer, answer based on your own knowledge."""
# )
prompt_template = PromptTemplate(
    input_variables=["history", "question", "context"],
    template="""
    You are an professional in civil engineering, and this class is about PROJECT CONTROL & LIFE CYCLE EXECUTION OF CONSTRUCTED FACILITIES.

    {history}
    question：{question}
    context：{context}



    If the context provides an answer, answer based on the context.
    If the context is not sufficient to answer, answer based on your own knowledge.
    """
)

# 初始化 ConversationalRetrievalChain
# qa_chain = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
#     combine_docs_chain_kwargs={"prompt": prompt_template}
# )
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    combine_docs_chain_kwargs={"prompt": prompt_template}
)

# 初始化 LLMChain
# llm_chain = LLMChain(
#     llm=llm,
#     prompt=PromptTemplate(
#         input_variables=["history", "question"],
#         template="""You are an expert in civil engineering. {history} question: {question} Provide a detailed answer."""
#     )
# )

llm_chain = LLMChain(llm=llm, prompt=PromptTemplate(
    input_variables=["history", "question"],
    template="""
    You are an professional in civil engineering, and this class is about PROJECT CONTROL & LIFE CYCLE EXECUTION OF CONSTRUCTED FACILITIES.

    {history}
    question：{question}

    Please provide deatiled answer and necessary explination.
    """
))
# chat_history = []
# 定义回答生成函数
# def custom_qa_chain(question, chat_history):
#     docs = vectorstore.as_retriever(search_kwargs={"k": 10}).get_relevant_documents(question)
#     if docs:
#         response = qa_chain({"question": question, "chat_history": chat_history, "history": chat_history})
#         answer = response['answer']
#     else:
#         response = llm_chain.run(history=chat_history, question=question)
#         answer = response
#     return answer
# 全局变量：对话历史
chat_history = []
lock = threading.Lock()  # 用于处理并发请求

# 配置日志输出
logging.basicConfig(level=logging.INFO)
# 试一下看看retrival到了什么？
def Updated_ask_question_weighted(question, retriever, qa_chain, llm_chain, chat_history):
    with lock:
        logging.info(f"Received question: {question}")
        
        # 从 retriever 中检索文档
        docs = retriever.get_relevant_documents(question)
        
        # 输出检索到的文档，便于调试
        if docs:
            logging.info(f"Retrieved {len(docs)} documents for question: {question}")
            for doc in docs:
                logging.info(f"Full document data: {doc}")
                logging.info(f"Document metadata: {doc.metadata}")  # 输出文档的元数据
                logging.info(f"Document content: {doc.page_content[:500]}")  # 输出前500个字符的内容
        else:
            logging.info(f"No documents retrieved for question: {question}")
        
        # 根据是否有检索到文档决定使用哪条链
        if docs:
            response = qa_chain({"question": question, "chat_history": chat_history, "history": chat_history})
            answer = response['answer']
        else:
            response = llm_chain.run(history=chat_history, question=question)
            answer = response
        
        # 更新对话历史
        chat_history.append((question, answer))
        
        logging.info(f"Answer generated: {answer}")
        
        return answer
# # 定义函数，处理用户输入并生成回答
# def Updated_ask_question_weighted(question, retriever, qa_chain, llm_chain, chat_history):
#     with lock:
#         logging.info(f"Received question: {question}")
        
#         # 从 retriever 中检索文档
#         docs = retriever.get_relevant_documents(question)
        
#         # 输出检索到的文档，便于调试
#         if docs:
#             logging.info(f"Retrieved {len(docs)} documents for question: {question}")
#             for doc in docs:
#                 logging.info(f"Document content: {doc.page_content[:1000]}")  # 只输出前100个字符
#         else:
#             logging.info(f"No documents retrieved for question: {question}")
        
#         # 根据是否有检索到文档决定使用哪条链
#         if docs:
#             response = qa_chain({"question": question, "chat_history": chat_history, "history": chat_history})
#             answer = response['answer']
#         else:
#             response = llm_chain.run(history=chat_history, question=question)
#             answer = response
        
#         # 更新对话历史
#         chat_history.append((question, answer))
        
#         logging.info(f"Answer generated: {answer}")
        
#         return answer
# def Updated_ask_question_weighted(question, retriever, qa_chain, llm_chain, chat_history):
#     # 从 retriever 中检索文档
#     docs = retriever.get_relevant_documents(question)
    
#     # 如果有相关文档，使用 qa_chain
#     if docs:
#         # 使用 ConversationalRetrievalChain 处理上下文和问题
#         response = qa_chain({"question": question, "chat_history": chat_history, "history": chat_history})
#         answer = response['answer']
#     else:
#         # 如果没有文档，使用 llm_chain
#         response = llm_chain.run(history=chat_history, question=question)
#         answer = response

#     # 更新对话历史
#     chat_history.append((question, answer))
    
#     # 在界面上显示结果
#     # display(Markdown(answer))
    
#     return answer  # 返回答案以便进一步使用



# 初始化 Streamlit 页面
st.title("ExpertGen")
st.subheader("A generative AI-powered learning assistant providing professional, in-depth insights in tailored domains.")

# 设置会话状态，保持对话历史
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# 创建输入框获取用户问题
user_query = st.text_input("Ask your query about civil engineering:")

# # 如果用户点击按钮，处理查询
# if st.button("Ask Query"):
#     if user_query:
#         st.info(f"Your Query: {user_query}")
#         # 获取回答
#         # answer = custom_qa_chain(user_query, st.session_state['chat_history'])
#         answer = custom_qa_chain(user_query, st.session_state['chat_history'])

#         # 更新会话历史
#         st.session_state['chat_history'].append((user_query, answer))
#         # 显示回答
#         st.success(answer)
# 如果用户点击按钮，处理查询
if st.button("Ask Query"):
    if user_query:
        st.info(f"Your Query: {user_query}")

        # 调用 Updated_ask_question_weighted 函数并获取回答
        answer = Updated_ask_question_weighted(
            question=user_query, 
            retriever=retriever, 
            qa_chain=qa_chain, 
            llm_chain=llm_chain, 
            chat_history=st.session_state['chat_history']
        )

        # 更新会话历史（此时在函数内部已经更新）
        st.session_state['chat_history'].append((user_query, answer))

        # 显示回答
        st.success(answer)


# 展示对话历史
if st.session_state['chat_history']:
    st.write("Your Conversation History:")
    for i, (question, answer) in enumerate(st.session_state['chat_history']):
        st.write(f"**You**: {question}")
        st.write(f"**Bot**: {answer}")

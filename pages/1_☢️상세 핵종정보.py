import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

def main():
    loader = TextLoader("Rad_Info.txt", encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    # llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0)

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0)

    prompt_template = """아래의 문맥을 사용하여 질문에 답하십시오.
    지식전달을 목적으로 전문적으로 말해주세요.
    정보를 찾을 수 없으면 DB에 자료가 없다고 말해주세요.
    {context}
    질문: {question}
    유용한 답변:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )

    st.title("상세 핵종정보[RAG]")

    # 사용자 입력
    question = st.text_input("정보를 얻기 원하는 핵종과 내용(반감기, 취급주의사항 등등)을 질문하세요.")

    if question:

        result = qa_chain(question)
        st.subheader("답변:")
        st.write(result["result"])

if __name__ == "__main__":
    main()

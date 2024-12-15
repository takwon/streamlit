import streamlit as st
import os
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader

def load_docs(fileName):
    loader = TextLoader(fileName, encoding='UTF-8')
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)

    return splits


def create_vectorstore(splits):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

def create_store_vectorstore(splits):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings,persist_directory='./')
    vectorstore.persist()

# 데이터가 저장되어 있는 데이터베이스 불러오기
def create_rag_chain(vectorstore):
    database = Chroma(
        embedding_function=OpenAIEmbeddings(),  # 텍스트 임베딩 모델
        persist_directory='./'  # 데이터베이스 경로
    )

    # 검색기 설정
    retriever = database.as_retriever(
        search_type="similarity",  # Cosine Similarity, 문서 간의 유사성을 측정하는 방법
        search_kwargs={"k": 3,} # 검색 결과 중 상위 3개만을 가져오도록 설정
    )

    # 모델 설정
    chat_model = ChatOpenAI(model="gpt-4o", streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
    # 스트리밍 모드를 활성화하여 결과를 실시간으로 받을 수 있도록 설정
    # 스트리밍 중 출력 결과를 처리하기 위한 콜백 핸들러 설정. 이 콜백은 결과를 표준 출력(stdout)으로 실시간으로 보냄


    # QA 체인 설정
    qa = RetrievalQA.from_llm(llm=chat_model, retriever=retriever, return_source_documents=True)

    return qa
    # llm model과 retriever 설정
    # 답변과 함께 원본 문서를 반환하도록 설정

    # QA 체인은 사용자가 질문을 입력하면, 해당 질문과 유사한 문서를 검색하고, 검색된 문서를 바탕으로 GPT-4 모델이 답변을 생성해준다!

##### 메인 함수 #####
def main():

    st.title("상세 핵종정보[RAG]")

    # 사용자 입력
    question = st.text_input("정보를 얻기 원하는 핵종과 내용(반감기, 취급주의사항 등등)을 질문하세요.")

    if question:
        #if st.button("답변 받기"):
            with st.spinner("처리 중..."):
                # 문서 로드 및 분할
                splits = load_docs('Radionuclides.txt')

                # 벡터 저장소 생성
                vectorstore = create_store_vectorstore(splits)

                # RAG 체인 생성
                qa_chain = create_rag_chain(vectorstore)

                # 질문에 대한 답변 생성
                result = qa_chain({"query": question})

                st.subheader("답변:")
                st.write(result["result"])

if __name__ == "__main__":
    main()

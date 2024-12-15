import streamlit as st
# OpenAI 패키지 추가
import openai

##### 기능 구현 함수 #####

def ask_gpt(prompt, model):


    completion = openai.chat.completions.create(
        model=model,
        messages=prompt,
        temperature=0.7,
    )

    return completion.choices[0].message


#system_role = "You are a helpful assistant. Answer in korea"
system_role = """
"당신은 방사선학 전문가이며, 방사선학 자격 시험 문제를 생성하는 역할을 맡고 있습니다. 사용자가 요청한 난이도에 따라 적절한 시험 문제를 생성하세요.
시험 문제 출제만 응답하여 문제 출제합니다. 당신이 난이도, 유형(객관식, 주관식)를 알지못하는 경우에는 사용자에게 다시 질문해 달라고 요청한다.
객관식의 경우에는 보기를 질문 다음부터 각각 한줄씩 띄워서 출력해줘. 마지막에는 정답과 함께 나타내줘.

- 초급(beginner): 단순한 질문 (정답이 하나로 명확한 질문)으로서 단일 정답을 요구하는 간단한 문제를 만드세요. 예를 들어, 방사선의 기본 개념, 방사성 동위원소의 기본 특성, 또는 단일 선택형 문제를 포함하세요.
- 중급(보통): 문제를 해결하기 위해 두 가지 이상의 정보를 결합해야 하거나 방사선 안전 기준, 선량 단위 변환 등을 다루는 문제를 만드세요.
고급(어려움): 다양한 접근 방식이나 결합된 답변을 요구하는 심화된 문제를 만드세요. 예를 들어, 방사선 치료 계획, 방사선 물리학의 응용, 또는 방사선 생물학적 영향 분석 등을 다루세요.
사용자 요청:

난이도: [초급/중급/고급]
생성할 문제 수: [숫자 입력]
사용자의 요청을 기반으로 적절한 난이도의 방사선학 시험 문제를 출력하세요. 각 문제는 명확하고 구조화되어야 하며, 초급에서 고급으로 갈수록 복잡성과 응답의 다양성이 증가해야 합니다.

예시 출력:

초급 문제 (쉬움):
질문: "코발트-60의 반감기는 몇 년입니까? \n\n"

A) 30년
B) 20년
C) 1일
D) 5.27년 \n\n
답변: 4번 5.27년

중급 문제 (보통):
질문: "100 mSv의 방사선량을 받은 조직의 위험 계수(weighting factor)가 0.05일 때, 조직에 부여된 유효선량은 얼마입니까? \n\n"
답변: "5 mSv"

고급 문제 (어려움):
질문: "환자의 폐암 치료를 위해 방사선 치료 계획을 세우고 있습니다. 선량 60 Gy를 전달하기 위해 6 MV 선형 가속기를 사용하는 경우, 다음 조건을 기반으로 치료 계획을 제안하세요:

방사선이 도달해야 하는 깊이는 10 cm입니다.
방사선 감쇠를 고려하여 피부 표면에서의 초기 선량을 계산하세요. \n\n"
답변: "문제의 조건에 따라 다양한 계산 및 접근 방법 가능 (사용자가 계산해야 함)."
"""
##### 메인 함수 #####
def main():

    # 제목
    st.header("면허시험 문제 출제자")
    # 구분선
    st.markdown("---")

    # 기본 설명
    with st.expander("면허시험 문제 출제", expanded=True):
        st.write(
            """     
            - 방사선 면허시험 대비 문제출제를 요구한다.
            - 난이도를 초급,중급,고급으로 나누어 요청한다.
            """
        )

        st.markdown("")

        # 사용자 입력
    question = st.text_input("방사선면허 문제출제를 난이도와 함께 요청하세요.")

    if question:
        with st.spinner("처리 중..."):
            messages = [
                {"role": "system", "content": system_role},
                {"role": "user", "content": question}
            ]
            response = ask_gpt(messages, model="gpt-4o").content
            st.write(response)

if __name__ == "__main__":
    main()


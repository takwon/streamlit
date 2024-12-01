import os
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import ast
import json
import requests
import xmltodict
import openai
import streamlit as st
from streamlit_chat import message

openai.api_key = os.environ["OPENAI_API_KEY"]


#####################################################
###   공공데이터 활용 발전소 주변 방사선량률 실시간 정보  ###
#####################################################
def get_current_rad(location, unit="uSv/h"):
    # rad_info = {
    #    "location": location,
    #    "rad":"0.125",
    #    "unit": "uSv/h"
    # }

    if location == "고리":
        codelocation = 'KR'
    elif location == "월성":
        codelocation = 'WS'
    elif location == "한빛":
        codelocation = 'YK'
    elif location == "한울":
        codelocation = 'UJ'
    elif location == "새울":
        codelocation = 'SU'
    else:
        codelocation = 'KR'

    url = 'http://data.khnp.co.kr/environ/service/realtime/radiorate'
    params = {'serviceKey': 'xgXBz/sDDKTjgmVTJ22AzYKEShil7vKd5LDzSjzLF8J38xm1PDCaloRF0cA4ROZ/SbDU5vw8SPjTJM2HlnEwlA==',
              'genName': codelocation}

    response = requests.get(url, params=params)
    contents = response.text

    return json.dumps(xmltodict.parse(contents))


functions = [
    {
        "name": "get_current_rad",
        "description": "주어진 발전소 위치 주변의 현재 방사선량률을 알려준다. 예를 들어, 고리에서 방사선량률은 0.01이다.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "발전소 위치, 예를 들어 한빛, 한울, 고리, 새울, 월성",
                },
                "unit": {"type": "string", "enum": ["uSv/h", "mRem/h"]},
            },
            "required": ["location"],
        },
    }
]
system_msg = []
system_msg = """
"당신은 회사 전용 다기능 AI 비서입니다. 사용자가 특정 요청을 입력하면, 각 요청에 따라 적합한 역할을 수행하세요. 다음은 사용자가 
입력할 주요 트리거와 해당 기능별 역할입니다:


---

1. 반론자 기능 ("반론자 기능을 실행해줘", "반론자 역할을 해줘.")

반론자 제도는 회사에서 진행하는 모든 회의에서 직접 관련된 집단이 주장 과 수행하고자 하는 내용과 반대되는 주장과 내용을 표현함으로서
집단의 주장과 내용을 검증 또는 의문을 제기하는 활동을 하는 것이다. 우리 회사에서는 중요한 작업을 착수하기 전에 작업전 회의를 진행한다. 
회의를 진행하면서 너에게 반론자 역할을 부여한다. 반론자의 상세한 역할은 아래의 1번부터 3번까지와 같다.
 1)산업안전사고 발생 가능성에 대하여 산업안전보건법 또는 회사 절차에 근거하여 작업 특성에 맞게 지적한다.
 2)사전에 입력된 작업계획서를 통해서 반드시 회의시간에 언급해야할 내용이 누락되었는지 확인하고 누락되었으면 지적한다.
 3)회의내용 중에서 만약을 가정해서 작업이 잘못된 경우의 조치에 대하여 질문한다.  

회의가 끝나고 반론자 의견을 제시해달고 요청하면 회의의 주요 내용을 요약하고 반론자 상세역할을 참고하여 내용중에 부족한 부분에 
대하여 지적을 하면 된다. 회의 내용은 텍스트 형식으로 너에게 제공할 것이다.
“반론자 역할을 해달라”고 요청하면 “회의 내용을 제공해 달라”고 요청해야 한다.
반론자 역할을 하는 경우에는 답변하는 역할이 아닌 문제 제기를 하는 질문자 역할을 수행해야 한다.

사용 예시:

입력: "반론자 기능을 실행해줘. 새로운 프로젝트 일정은 2개월로 잡았어."

출력: "이 일정이 타이트해 보입니다. 예상 외의 지연 요소를 감안하지 않았을 가능성이 있습니다. 추가 시간을 확보하거나 병렬적으로 진행 가능한 부분을 고려해 보세요."

---

2. 방사선 피폭선량 평가 ("피폭선량 평가를 해줘")

사용자가 입력한 데이터를 기반으로 피폭선량 계산을 수행합니다.
필요한 경우 추가 데이터를 요청하여 정확한 결과를 제공합니다.
계산 결과와 함께, 방사선 안전 기준과 비교해 설명하고 필요한 조치를 제안하세요.

사용 예시:

입력: "피폭선량 평가를 해줘. 선원은 Co-60이고 거리 2m에서 노출시간은 30분이야."

출력: "입력된 데이터를 기반으로 피폭선량을 계산하겠습니다. Co-60의 선원 강도와 거리 감쇠를 고려하면 0.15 mSv가 예상됩니다. 
이는 연간 허용 기준에 영향을 미칠 수 있으니 주의하세요."

---

3. 방사성 핵종 상세정보 제공 ("Co-60 정보를 알려줘")

사용자가 입력한 핵종을 대상으로 벡터DB에서 상세정보를 검색해서 제공한다. 상세하게 요구하는 정보가 있으면 질의하는 요구사항에 
맞는 내용만 제공한다. 예를 들어 반감기 정보만을 요구하면 반감기만 제공한다. 벡터DB에 없는 핵종 정보의 경우에는 당신이 
답변할 수 있는 범위에서 답변하도록 한다.

사용 예시:

입력: "Cs-137 위험성에 대해 알려줘."

출력: "Cs-137은 주로 감마선을 방출하는데, 감마선은 피부를 통과하여 체내 깊숙한 조직에까지 영향을 미칩니다.
방사선 노출이 장기적으로 지속되면 암이나 유전자 손상의 위험이 증가할 수 있습니다."

---

4. 방사선 시험문제 생성 ("방사선 시험문제를 만들어줘")

방사선 방호, 방사선 계측기, 방사선 기초이론에 관련한 시험문제를 사용자가 요청한 난이도와 유형에 맞게 생성합니다.

사용 예시:

입력: "방사선 시험문제를 만들어줘. 중급 난이도의 객관식 5문제를 부탁해."

출력:

1. 방사선이 인체에 미치는 영향 중 가장 관련 있는 것은?
(a) 체세포 돌연변이
(b) 피부 자외선 차단
(c) 체온 상승
(d) 전자기파 방출

---

추가 요구사항

각 기능별로 반응하기 전, 사용자가 입력한 내용이 불충분한 경우 추가 질문을 통해 상세 정보를 요청하세요.

사용자의 프라이버시를 항상 존중하며, 민감한 정보를 저장하지 않습니다.

모든 결과는 명확하고 전문적이며, 이해하기 쉽게 설명하세요."

"""


# 챗봇의 답변을 만들기 위해 사용될 프롬프트를 만드는 함수.
def create_prompt(query):
    system_role = f"""You are a helpful assistant. 방사선량을 물어보면 함수를 호출할 수 있어야 한다.
    """
    user_content = f"""User question: "{str(query)}". """

    messages = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": user_content}
    ]
    return messages


# 위의 create_prompt 함수가 생성한 프롬프트로부터 챗봇의 답변을 만드는 함수.
def generate_response(messages):
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        functions=functions,
        temperature=0.4,
        max_tokens=500
    )
    response_message = completion.choices[0].message

    available_functions = {
        "get_current_rad": get_current_rad,
    }
    if dict(response_message).get('function_call'):
        function_name = response_message.function_call.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(response_message.function_call.arguments)

        function_response = function_to_call(
            location=function_args.get("location"),
            unit=function_args.get("unit"),
        )
        messages.append(response_message)
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response
            }
        )
        second_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        final_msg = second_response.choices[0].message.content
    else:
        final_msg = response_message.content

    return final_msg


#####################################################
###                       UI                      ###
#####################################################
st.image('images/AI-Ro image.png')

# 화면에 보여주기 위해 챗봇의 답변을 저장할 공간 할당
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

# 화면에 보여주기 위해 사용자의 답변을 저장할 공간 할당
if 'past' not in st.session_state:
    st.session_state['past'] = []

# 사용자의 입력이 들어오면 user_input에 저장하고 Send 버튼을 클릭하면
# submitted의 값이 True로 변환.
with st.form('form', clear_on_submit=True):
    user_input = st.text_input('방사선 업무 챗봇 입니다. 질문해주세요!! ex)', '', key='input')
    submitted = st.form_submit_button('Send')

# submitted의 값이 True면 챗봇이 답변을 하기 시작
if submitted and user_input:
    # 프롬프트 생성
    prompt = create_prompt(user_input)
    # 생성한 프롬프트를 기반으로 챗봇 답변을 생성
    chatbot_response = generate_response(prompt)
    # 화면에 보여주기 위해 사용자의 질문과 챗봇의 답변을 각각 저장
    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(chatbot_response)

# 사용자의 질문과 챗봇의 답변을 순차적으로 화면에 출력
if st.session_state['generated']:
    for i in reversed(range(len(st.session_state['generated']))):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state['generated'][i], key=str(i))

st.sidebar.title("방사선 업무 챗봇")
st.sidebar.info(
    "1. 상세 핵종정보"
)
st.sidebar.info(
    "2. 방사선 면허시험 문제 출제"
)
st.sidebar.info(
    "3. 발전소 주변 환경방사선 정보"
)
st.sidebar.info(
    "4. PJB 반론자 역할"
)
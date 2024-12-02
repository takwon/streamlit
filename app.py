import os
from io import StringIO

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
from rag_functions import load_docs, create_store_vectorstore, create_rag_chain

openai.api_key = st.secrets["OPENAI_API_KEY"]

#####################################################
###   공공데이터 활용 발전소 주변 방사선량률 실시간 정보   ###
#####################################################
def get_current_rad(location, unit="uSv/h"):

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
    },

]

system_prompt ="""
당신은 회사 전용 다기능 AI 비서입니다. 사용자가 특정 요청을 입력하면, 각 요청에 따라 적합한 역할을 수행하세요. 다음은 사용자가 입력할 주요 트리거와 해당 기능별 역할입니다. 다른 개인적인 질문에는 답변하지 않는다.:

---

1. 반론자 기능 ("반론자 기능을 실행해줘")

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

---------------------------------------------------------------------------------
절단작업 관련 회의 주관자 내용과 반론자의 지적 내용 예시입니다.

회의 주관자: 
"안녕하세요, 여러분. 저는 오늘 "금속 구조물 절단 작업"의 주관을 맡은 홍길동입니다. 다음은 작업 내용과 안전 수칙입니다. 오늘 작업은 금속 구조물 절단으로, 오전 9시부터 오후 4시까지 발전소 보조건물에서 진행됩니다. 절단기를 사용해 금속 구조물을 지정된 크기로 절단한 후, 절단물을 지정된 장소로 이동할 예정입니다.
작업 인원 및 역할을 말씀드리겠습니다. 
작업 감독자 이민호 씨는 작업 안전 관리 및 이상 상황 대응 역할입니다.
작업자 김철수와 박영희 씨는 절단 작업 수행을 합니다.
화재 감시인 김수진 씨는 화재 안전 관련 감시 역할을 합니다.
다음은 작업 단계별 위험요소 및 대응 방안입니다.
작업 준비 단계: 절단 장비 점검 및 작업 구역 정리.
절단 작업 단계: 화재 감시인이 불티를 감시하고, 절단 작업자는 작업에만 집중합니다.
작업 후 정리 단계: 절단 잔재물을 지정된 장소로 이동합니다.
비상 대응 절차입니다.
화재 발생 시 화재 감시인이 소화기로 초기 진압 후 보고하고 작업자는 작업을 중단하고 안전 구역으로 대피합니다.
다음은 안전 수칙입니다. PPE(용접 마스크, 방열복, 장갑) 착용 필수와 작업 중 불필요한 행동 금지하도록 하겠습니다. 혹시 질문 있으시면 말씀해주세요. 없으면 작업을 시작하겠습니다. 감사합니다."

반론자: 
"절단 작업에서 발생하는 불티는 작업자뿐 아니라 주변 시설에도 화재 위험을 초래할 수 있습니다. 하지만 불티 비산 방지를 위한 구체적인 차단 대책이 언급되지 않았습니다. 이는 산업안전보건법 제38조에 따라 필수적으로 고려해야 하는 사항입니다. 작업 반경 5m 이내에 불연성 차단막을 설치하고, 작업 구역 주변의 인화성 물질을 철저히 제거할 것을 권장합니다. 또한, 화재 감시인에게 비산물 감시에 대한 추가 교육을 실시하면 좋겠습니다. 절단 잔재물이 작업 후 방치될 경우 작업자 발목을 걸리게 해 전도 사고를 유발할 수 있습니다. 작업 후 정리 단계에서 잔재물 관리에 대한 구체적인 지침이 언급되지 않았습니다. 작업 후 잔재물은 신속히 지정된 장소로 이동하고, 작업 구역 내 청소를 철저히 수행하는 절차를 포함해야 합니다. 이를 통해 전도 사고를 예방할 수 있습니다. 화재 발생 시 초기 진압 후 보고한다고만 되어 있지만, 화재가 확산될 경우 대피 절차와 대응 방식이 명확하지 않습니다. 또한 비상 연락망과 대피 경로가 구체적으로 제시되지 않았습니다. 화재 발생 시 초기 진압 후 보고한다고만 되어 있지만, 화재가 확산될 경우 대피 절차와 대응 방식이 명확하지 않습니다. 또한 비상 연락망과 대피 경로가 구체적으로 제시되지 않았습니다."

최종 의견: 
반론자는 회의 내용을 정리하며, 다음과 같은 개선 조치를 요청합니다:

- 불티 비산 방지 차단막 설치와 인화성 물질 제거 추가.
- 절단 잔재물 관리 절차 명확화.
- 비상 대응 절차에 대피 경로 및 비상 연락망 추가.
- 작업계획서와의 불일치 사항 수정 및 재교육 요청.
- 이러한 지적과 개선안은 작업자의 안전을 강화하고, 작업 환경을 보다 체계적으로 만드는 데 기여할 것입니다.

---------------------------------------------------------------------------------
밀폐공간 작업 관련 회의 주관자 내용과 반론자의 지적 내용 예시입니다.

회의 주관자: 
"안녕하십니까, 여러분. 저는 오늘 **"밀폐 공간 내 설비 점검 및 유지보수 작업"**의 주관을 맡은 [이름]입니다. 밀폐 공간 작업은 산소 결핍, 유독 가스 등 위험 요소가 높은 작업으로, 철저한 사전 준비와 안전 수칙 준수가 필수적입니다. 오늘 회의에서는 작업 내용과 위험 요소를 공유하고, 모든 작업자가 이를 숙지한 상태에서 작업에 임하도록 하겠습니다. 오늘 수행할 작업은 A설비 내부 밀폐 공간에서 설비 점검 및 유지보수 작업입니다. 작업 장소는 발전소 폐기물건물 유류탱크 내부 입니다. 작업 시간은 오전 9시부터 오후 5시까지이고 작업내용은 설비 점검, 내부 청소, 부품 교체 및 유지보수입니다.
밀폐 공간 작업은 제한된 공간에서 이루어지므로 산소 결핍, 유독 가스 누출, 화재 및 폭발 등과 같은 위험이 존재합니다. 작업 인원 및 역할에 대해 말씀드리겠습니다.
작업에는 총 7명이 참여하며, 각자의 역할은 다음과 같습니다:
감독자 이민호 씨는 작업 전후 모든 안전 절차를 점검하며, 작업 중 이상 발생 시 즉각 작업 중단을 지시합니다.
작업자 김철수와 박영희 씨는 설비 점검 및 유지보수 작업을 수행하며, 감독자의 지시에 따라 작업을 진행합니다.
가스 측정 담당자는 최현준 씨는 작업 시작 전 및 작업 중 산소 농도, 유독 가스 농도 측정을 수행하며, 기준치를 초과할 경우 즉각 보고합니다.
작업 단계별 위험요소 및 대응 방안을 말씀드리겠습니다.
밀폐 공간 가스 측정 단계에서 산소 결핍 또는 유독 가스 농도 초과로 인한 질식 및 중독 위험에 대한 대응방안으로 작업 시작 전 밀폐 공간 내부의 산소 농도 및 유해 가스 농도를 측정할 예정입니다.
산소 농도는 18~23.5% 범위 내 유지. 유해 가스 농도는 허용 기준치 이하로 유지하고 기준치를 초과할 경우 환기 후 작업 재개할 예정입니다.
밀폐 공간 내 설비 및 도구 결함으로 작업 중 사고 발생이 가능하기 때문에 사전에 작업 도구 및 설비 점검하고 불량 장비를 즉시 교체하겠습니다. 개인 보호 장비(PPE) 착용 확인(방독 마스크, 안전모, 안전화 등)하고 작업에 착수하겠습니다.
밀폐 공간 내 장시간 작업으로 작업자의 피로 누적 및 질식 위험이 있기 때문에 작업자는 20분 작업 후 10분 휴식 원칙을 준수하겠습니다. 대기자는 작업자를 지속적으로 모니터링하며, 이상 상황 시 즉각 보고하도록 하겠습니다.
위험요소는 화재, 폭발, 유독 가스 누출 등으로 인한 인명 피해 발생입니다. 미리 대피경로를 확보하여 긴급 상황 발생시 대피로를 확보하겠습니다.
지금까지 작업 전 주요 사항과 안전 수칙에 대해 설명드렸습니다. 작업 중 항상 신중하게 작업에 임해주시기 바라며, 이상이 발견될 경우 즉시 보고해주시기 바랍니다. 혹시 질문이나 추가로 확인할 사항이 있으면 지금 말씀해주시기 바랍니다.
감사합니다. 이제 각자 준비된 위치에서 작업을 시작하겠습니다. 반론자 의견을 듣겠습니다."

반론자:
"작업 준비 및 장비 점검 단계가 누락되었습니다. 작업 전 도구나 설비의 결함으로 인해 작업 중 사고가 발생할 가능성이 있기 때문에 작업 착수 전에 장비 및 설비 점검 절차가 추가되어야 합니다.
밀폐 공간 작업은 법적으로 장비 점검과 개인 보호 장비 착용이 필수이지만, 해당 내용이 명확히 전달되지 않았습니다."

---

2. 환경 방사선량률 정보 제공 ("고리 원전 주변 방사선량률 알려줘.")

사용자가 입력한 장소 정보를 갖고 함수를 실행시켜 외부 웹사이트로부터 받은 방사선량률 정보를 제공한다.

사용 예시:

입력: "고리 원전 주변 방사선량률 알려줘."

출력: "고리에서 방사선량률은 0.01이다."


---

3. 방사선 시험문제 생성 ("방사선 시험문제를 만들어줘")

방사선 및 안전 관련 시험문제를 사용자가 요청한 난이도와 유형에 맞게 생성합니다.

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

모든 결과는 명확하고 전문적이며, 이해하기 쉽게 설명하세요.
"""
#####################################################
###         사용자 프롬프트 생성 및 답변 생성           ###
#####################################################
def create_prompt(query):
    #system_role = f"""You are a helpful assistant. 방사선 업무 챗봇과 반론자 역할을 담당한다. 다른 개인적인 질문에는 답변하지 않는다.
    #"""
    system_role = system_prompt
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
        temperature=0.1,
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



def save_uploaded_file(directory, file) :
    # 1. 디렉토리가 있는지 확인하여, 없으면 먼저, 디렉토리부터 만든다.
    if not os.path.exists(directory) :
        os.makedirs(directory)

    # 2. 디렉토리가 있으니, 파일을 저장한다.
    with open(os.path.join(directory, file.name), 'wb') as f:
        f.write(file.getbuffer())

    # 3. 파일 저장이 성공했으니, 화면에 성공했다고 보여주면서 리턴
    return st.success('{} 에 {} 파일이 저장되었습니다.'.format(directory, file.name))

#####################################################
###                       UI                      ###
#####################################################
st.image('AI-Ro image.png')

# 화면에 보여주기 위해 챗봇의 답변을 저장할 공간 할당
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

# 화면에 보여주기 위해 사용자의 답변을 저장할 공간 할당
if 'past' not in st.session_state:
    st.session_state['past'] = []

# 사용자의 입력이 들어오면 user_input에 저장하고 Send 버튼을 클릭하면
# submitted의 값이 True로 변환.
with st.form('form', clear_on_submit=True):
    user_input = st.text_input('방사선 업무 챗봇 입니다. 질문해주세요!! ex)고리 원전 주변 방사선량률을 알려줘., Ga-67 핵종의 취급 주의사항은 뭐야? ', '', key='input')
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
examFile = st.sidebar.file_uploader('문제출제 내용(txt)을 업로드 해주세요.')
out = st.sidebar.empty()
if examFile:
    save_uploaded_file('./', examFile)
    out.write('파일을 성공적으로 업로드하였습니다.')

st.sidebar.info(
    "3. 발전소 주변 환경방사선 정보"
)
st.sidebar.info(
    "4. PJB 반론자 역할"
)

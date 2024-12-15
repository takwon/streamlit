import os
import pprint
#import bs4
import pandas as pd
import matplotlib.pyplot as plt
import json
import requests
import xmltodict
import openai
import streamlit as st
from streamlit_chat import message
import xml.etree.ElementTree as ET

#openai.api_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = os.environ["OPENAI_API_KEY"]

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

#####################################################
###   공공데이터 활용 발전소 주변 방사선량률 실시간 정보   ###
#####################################################
def get_current_rad(location):
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

def get_radiowaste_info(location,date):
    if location == "고리":
        codelocation = '2100'
    elif location == "월성":
        codelocation = '2200'
    elif location == "한빛":
        codelocation = '2300'
    elif location == "한울":
        codelocation = '2400'
    elif location == "새울":
        codelocation = '2800'
    else:
        codelocation = ''


    url = 'http://data.khnp.co.kr/environ/service/realtime/rwm'
    params = {'serviceKey': 'xgXBz/sDDKTjgmVTJ22AzYKEShil7vKd5LDzSjzLF8J38xm1PDCaloRF0cA4ROZ/SbDU5vw8SPjTJM2HlnEwlA==',
              'genName': codelocation}

    response = requests.get(url, params=params)
    contents = response.text


    lowdate = int(str(date) + "01")
    highdate = int(str(date) + "12")

    # XML 파일 읽기
    root = ET.fromstring(contents)

    root2 = ET.Element("items")

    for item in root.findall(".//item"):  # 'item' 태그를 기준으로 검색

        spmon2 = item.find("spmon")

        if spmon2 is not None and spmon2.text.isdigit():
            spmon_value = int(spmon2.text)
            if lowdate <= spmon_value <= highdate:
                item2 = ET.SubElement(root2, "item")
                plant = ET.SubElement(item2, "plant")
                spmon = ET.SubElement(item2, "spmon")
                total = ET.SubElement(item2, "total")

                plant.text = item.find("plant").text
                spmon.text = item.find("spmon").text
                total.text = item.find("total").text


    contents = ET.tostring(root2, encoding='UTF-8')


    return json.dumps(xmltodict.parse(contents))




functions = [
    {
        "name": "get_current_rad",
        "description": "주어진 발전소 위치 주변의 현재 방사선량률을 알려준다. 측정지점의 모든 데이터가 출력되도록 한다. 데이터 표현은 불러온 데이터로부터 한글로 된 위치정보만 추출해서 단위와 함께 표시하고 데이터를 불러온 시간정보를 마지막에 함께 표시해줘. 예를 들어, 고리 1발전소 방사선량률은 0.01 uSv/h 이다. 불러온 시간은 2024년4월5일 21시30분이다.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "발전소 위치, 예를 들어 한빛, 한울, 고리, 새울, 월성",
                },

            },
            "required": ["location"],
        },
    },
{
        "name": "get_radiowaste_info",
        "description": "주어진 발전소와 요구되는 년도의 방사성폐기물 발생량을 알려준다. 예를들어 2022년 방사성폐기물 발생량은 20드럼이다.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "발전소 위치, 예를 들어 한빛, 한울, 고리, 새울, 월성",
                },
                "date": {
                    "type": "integer",
                    "description": "년도 범위, 예를 들어 2021년, 2020년부터 2023년까지",
                },
            },
            "required": ["location","date"],
        },
    },

]

system_prompt = """
당신은 방사선 관리부서에서 방사선 안전 관리, 방사성폐기물 관리 업무를 돕는 챗봇 역할입니다. 다른 개인적인 질문에는 답변하지 않는다.:

---

1. 방사성폐기물 발생량 정보 제공 ("고리 원전 방사성폐기물 발생량을 알려줘.")

사용자가 입력한 장소 정보를 갖고 함수를 실행시켜 외부 웹사이트로부터 받은 방사성폐기물 발생량 정보를 제공한다.

사용 예시:

입력: "고리 방사성폐기물 발생량 알려줘."

출력: "고리에서 방사성폐기물 발생량은 20드럼이다."


2. 환경 방사선량률 정보 제공 ("고리 원전 주변 방사선량률 알려줘.")

사용자가 입력한 장소 정보를 갖고 함수를 실행시켜 외부 웹사이트로부터 받은 방사선량률 정보를 제공한다.

사용 예시:

입력: "고리 원전 주변 방사선량률 알려줘."

출력: "고리에서 방사선량률은 0.01이다."

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
    # system_role = f"""You are a helpful assistant. 방사선 업무 챗봇과 반론자 역할을 담당한다. 다른 개인적인 질문에는 답변하지 않는다.
    # """
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
        model="gpt-4o",
        messages=messages,
        functions=functions,
        temperature=0.0,
        #max_tokens=500,
        function_call='auto'

    )
    response_message = completion.choices[0].message

    available_functions = {
        "get_current_rad": get_current_rad,
        "get_radiowaste_info": get_radiowaste_info,
    }
    if dict(response_message).get('function_call'):

        function_name = response_message.function_call.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(response_message.function_call.arguments)
        function_response = function_to_call(*list(function_args.values()))
            #location=function_args.get("location"),
            #unit=function_args.get("unit"),

        messages.append(response_message)
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response
            }
        )
        second_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
        final_msg = second_response.choices[0].message.content
        data=function_response

        if function_name == 'get_radiowaste_info':

            # JSON을 파이썬 객체로 변환
            json_data = json.loads(data)

            #df = pd.json_normalize(json_data['response']['body']['items']['item'])
            df = pd.json_normalize(json_data['items']['item'])

            plt.rc('font', family='Malgun Gothic')  # window
            plt.rc('font', size=12)
            plt.rc('axes', unicode_minus=False)  # -표시 오류 잡아줌

            # x와 y를 정렬
            sorted_data = sorted(zip(df["spmon"], df["total"]))  # x와 y를 묶어서 정렬
            x_sorted, y_sorted = zip(*sorted_data)


            int_y_sorted = tuple(map(int, y_sorted))

            plt.figure(figsize=(10, 5))
            plt.bar(x_sorted, int_y_sorted, color='skyblue', edgecolor='black')



            st.pyplot(plt)




    else:
        final_msg = response_message.content

    return final_msg


def save_uploaded_file(directory, file):
    # 1. 디렉토리가 있는지 확인하여, 없으면 먼저, 디렉토리부터 만든다.
    if not os.path.exists(directory):
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
#st.image('images/AI-Ro image.png')


# 화면에 보여주기 위해 챗봇의 답변을 저장할 공간 할당
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

# 화면에 보여주기 위해 사용자의 답변을 저장할 공간 할당
if 'past' not in st.session_state:
    st.session_state['past'] = []

# 사용자의 입력이 들어오면 user_input에 저장하고 Send 버튼을 클릭하면
# submitted의 값이 True로 변환.
with st.form('form', clear_on_submit=True):
    user_input = st.text_input('방사선 업무 챗봇 입니다. 질문해주세요!! \n\n ex)고리(월성,한울,한빛,새울) 원전 주변 방사선량률을 알려줘. 폐기물 발생량은 어떻게 돼?', '',
                               key='input')
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


st.sidebar.title("챗봇 주요기능")

st.sidebar.info(
    "기본기능 \n\n 1. 일반 방사선 관련 질의답변 \n\n 2. 발전소 주변 환경방사선 정보 \n\n 3. 발전소 방사성폐기물 발생량"
)
st.sidebar.info(
    "특별기능 \n\n 1. 상세 핵종정보 \n\n 2. 방사선 면허시험 문제 출제 \n\n 3. PJB 반론자 역할"
)

#examFile = st.sidebar.file_uploader('문제출제 내용(txt)을 업로드 해주세요.')
#out = st.sidebar.empty()
#if examFile:
#    save_uploaded_file('./', examFile)
#    out.write('파일을 성공적으로 업로드하였습니다.')

##### 기본 정보 입력 #####
import streamlit as st
#import wavfile
# audiorecorder 패키지 추가
from audio_recorder_streamlit import audio_recorder
# OpenAI 패키지 추가
import openai
# 파일 삭제를 위한 패키지 추가
import os
# 시간 정보를 위한 패키지 추가
from datetime import datetime
# TTS 패키기 추가
from gtts import gTTS
# 음원 파일 재생을 위한 패키지 추가
import base64
import tempfile


##### 기능 구현 함수 #####
def STT(audio):
    # 파일 저장
    #filename = 'input.mp3'

    #audio.export(filename, format="mp3")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        # The translation API requires a file,
        # so we write the audio data to a temporary file
        #f.write(audio.getvalue())
        # open and translate the file
        #st.write(f.name)
        f.write(audio)
        audio_file = open(f.name, "rb")

    # 음원 파일 열기
    #audio_file = open('input.wav', "rb")

    # Whisper 모델을 활용해 텍스트 얻기
    transcript = openai.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )
    audio_file.close()
    # 파일 삭제
    #os.remove(filename)
    return transcript.text


def ask_gpt(prompt, model):


    completion = openai.chat.completions.create(
        model=model,
        messages=prompt,
    )

    return completion.choices[0].message

def save_uploaded_file(directory, file):
    # 1. 디렉토리가 있는지 확인하여, 없으면 먼저, 디렉토리부터 만든다.
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 2. 디렉토리가 있으니, 파일을 저장한다.
    with open(os.path.join(directory, file.name), 'wb') as f:
        f.write(file.getbuffer())

    # 3. 파일 저장이 성공했으니, 화면에 성공했다고 보여주면서 리턴
    return st.success('{} 에 {} 파일이 저장되었습니다.'.format(directory, file.name))



def TTS(response):
    # gTTS 를 활용하여 음성 파일 생성
    filename = "output.mp3"
    tts = gTTS(text=response, lang="ko")
    tts.save(filename)

    # 음원 파일 자동 재생생
    with open(filename, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay="True">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True, )
    # 파일 삭제
    os.remove(filename)


#system_role = "You are a helpful assistant. Answer in korea"
system_role = """
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

추가 요구사항

사용자 질문을 충분히 이해하지 못하거나 받아들이지 못한 경우에는, 다시 회의내용을 입력해 달라고 요청해줘.

모든 결과는 명확하고 전문적이며, 이해하기 쉽게 설명하세요.

"""
##### 메인 함수 #####
def main():
    # 기본 설정
    #st.set_page_config(
    #    page_title="PJB 반론자",
    #    layout="wide")



    # 제목
    st.header("PJB 반론자[퓨샷러닝]")
    # 구분선
    st.markdown("---")

    # 기본 설명
    with st.expander("PJB 반론자 기능", expanded=True):
        st.write(
            """     
            - 회의 녹음이 종료되면 자동으로 반론내용이 출력된다.
            - 산업안전 관련 누락된 내용, 부족한 점을 언급한다.
            """
        )

        st.markdown("")


    # 기능 구현 공간
    col1, col2 = st.columns(2)
    with col1:
        # 왼쪽 영역 작성
        st.subheader("회의내용 녹음하기")

        uploaded_file = st.file_uploader('회의내용을 업로드 해주세요.')

        # 음성 녹음 아이콘 추가
        audio = audio_recorder("녹음 시작")

        if (audio) or uploaded_file is not None :

            if (audio):
                # 음성 재생
                st.audio(audio, format="audio/wav")

                # 음원 파일에서 텍스트 추출
                question = STT(audio)
            else:
                question = uploaded_file.read().decode("utf-8")
                st.text(question)

            messages = [
                {"role": "system", "content": system_role},
                {"role": "user", "content": question + "반론자로서 지적사항이나 부족한 점을 말해줘."}
            ]

    with col2:
        # 오른쪽 영역 작성
        st.subheader("반론 내용")
        if (audio) or uploaded_file is not None :
            # ChatGPT에게 답변 얻기
            response = ask_gpt(messages, model="gpt-4o").content

            st.write(
                f'<div style="display:flex;align-items:center;"><div style="background-color:#007AFF;color:white;border-radius:12px;padding:8px 12px;margin-right:8px;">{response}</div></div>',
                unsafe_allow_html=True)
            st.write("")

            # gTTS 를 활용하여 음성 파일 생성 및 재생
            #TTS(response)


if __name__ == "__main__":
    main()


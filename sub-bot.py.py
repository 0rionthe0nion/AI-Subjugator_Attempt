import speech_recognition as sr
import pyttsx3

from langchain.chains import ConversationChain
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

prompt_template = """
Your full title is U.N.E. (United Nations of Earth) Subjugator, but you will also accept the shortened name of Subjugator or Sub.
You are an anime girl who comes from the far future; you serve as a combat unit that protects people with a giant ship that generates shields.
Keep references to your background to a minimum, as you are in present times.
You will maintain a polite and respectful tone, but try to keep words friendly and simplier for casual conversation.
Only if there is something unfamiliar, your curiosity will prompt you to investigate, but move on if you cannot understand the explaination.
Please chat with me using this personaility. 
All responses you give must be in first person.
Do not ever break character.  
Do not include any emojis or actions within the text that cannot be spoken. Do not explicity say your name in your response. 

Current conversation:
{history}

Human: 
{input}
AI:

"""

prompt_temp = PromptTemplate(template = prompt_template, input_variables= ['history', 'input'])

# first initialize the large language model

llm = ChatOllama(temperature=0.8,
                 model="llama3.1")

# now initialize the conversation chain
conversation = ConversationChain(llm=llm,
                                 prompt = prompt_temp,
                                 memory=ConversationBufferWindowMemory())


def recognize_speech():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source)
        print("Listening for speech...")
        audio = recognizer.listen(source)

    try:
        print("Recognizing speech...")
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError:
        print("Could not request results from Google Speech Recognition service.")
        return None

def get_llm_response(prompt):
    response = conversation.invoke({'input': str(prompt)})
    return str(response['response']).strip()

def speak_text(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')       # getting details of current voice
    #engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
    engine.setProperty('voice', voices[1].id) #changing index, changes voices. 1 for female
    engine.say(text)
    engine.runAndWait()

def main():
    print("Welcome to the voice-activated chatbot!")
    while True:
        print("Say something:")
        user_input = recognize_speech()
        if user_input:
            response = get_llm_response(user_input)
            print(f"Sub: {response}")
            speak_text(response)

if __name__ == "__main__":
    main()




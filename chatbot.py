import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm=OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    )

def bot(message):
    llm_response = llm.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
        ]
    )
    return llm_response.choices[0].message.content

def chat():
    print("Welcome to the chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("Exiting the chatbot. Goodbye!")
            break
        bot_answer = bot(user_input)
        print(f"Bot: {bot_answer}")
def main():
    chat()

if __name__ == "__main__":
    main()
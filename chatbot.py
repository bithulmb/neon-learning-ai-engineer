import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm=OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    )

messages = [
    {
        "role": "system",
        "content": """
            You are a helpful assistant.
            Your name is Neon_bot.
            Always introduce yourself in the first message. 
            Give short and helpful answers.
            """
    }
]


total_tokens = 0

def bot(message):
    global total_tokens
    messages.append({"role": "user", "content": message})
    llm_response = llm.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages
    )

    response_content = llm_response.choices[0].message.content
    messages.append({"role": "assistant", "content": response_content})
    total_tokens += llm_response.usage.total_tokens
    return response_content

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
    print(f"Total tokens used in this session: {total_tokens}")

if __name__ == "__main__":
    main()
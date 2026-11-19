import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class Chatbot:
    def __init__(self):
        self.llm = OpenAI(
            api_key=os.getenv('GROQ_API_KEY'),
            base_url='https://api.groq.com/openai/v1'
        )

        self.messages = [
            {'role': 'system', 
             'content': '''
                        You are a helpful assistant.
                        Your name is neon_bot. in the first response, introduce yourself. 
                        give short and concise answers
                        '''
            }
        ]

        self.total_tokens = 0

    def bot(self,message):
 
        self.messages.append({"role": "user", "content": message})
        llm_response = self.llm.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=self.messages
        )

        response_content = llm_response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": response_content})
        self.total_tokens += llm_response.usage.total_tokens
        return response_content

    def chat(self):
        print("Welcome to the chatbot! Type 'exit' to quit.")
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == 'exit':
                print("Exiting the chatbot. Goodbye!")
                break
            bot_answer = self.bot(user_input)
            print(f"Bot: {bot_answer}")

    def report(self):
        for message in self.messages:
            role = message['role']
            content = message['content']
            print(f"{role.capitalize()}: {content}")
        print(f"Total tokens used: {self.total_tokens}")


def main():
    chatbot = Chatbot()
    chatbot.chat()
    chatbot.report()

if __name__ == "__main__":
    main()
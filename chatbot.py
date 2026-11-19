import os
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

from transformers import AutoTokenizer

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
        self.input_tokens = 0
        self.output_tokens = 0

    def bot(self,message):
 
        self.messages.append({"role": "user", "content": message})
        llm_response = self.llm.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=self.messages
        )

        response_content = llm_response.choices[0].message.content
        
        self.messages.append({"role": "assistant", "content": response_content})
        
        self.total_tokens += llm_response.usage.total_tokens
        self.input_tokens += llm_response.usage.prompt_tokens
        self.output_tokens += llm_response.usage.completion_tokens
        
        return response_content

    def chat(self):

        #tokenizer from tiktoken
        # encoding = tiktoken.get_encoding("cl100k_base")
        encoding = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
        
        print("Welcome to the chatbot! Type 'exit' to quit.")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'exit':
                print("Exiting the chatbot. Goodbye!")
                break

            token_count = len(encoding.encode(user_input))

            if token_count > 10:
                print("Input exceeds the maximum token limit of 10. Please enter a shorter message.")
                continue

            if self.total_tokens > 100:
                print("Total token usage exceeds the limit of 100. Please start a new session.")
                break
            
            print(f"User input tokens: {token_count}")
            
            bot_answer = self.bot(user_input)
            print(f"Bot: {bot_answer}")

    def report(self):
        for message in self.messages:
          print(message)
        
        print(f"Total tokens used: {self.total_tokens}")
        print(f"total cost: ${self.input_tokens * 0.0005 + self.output_tokens * 0.0008:.6f}")



def main():
    chatbot = Chatbot()
    chatbot.chat()
    chatbot.report()

if __name__ == "__main__":
    main()
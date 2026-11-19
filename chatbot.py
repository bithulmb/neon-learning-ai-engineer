import os
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken
import time

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

    def trim_messages(self,max_number=4):
        if len(self.messages) < max_number:
            return

        system_messages = [msg for msg in self.messages if msg['role'] == 'system']
        other_messages = [msg for msg in self.messages if msg['role'] != 'system']

        other_messages = other_messages[-max_number:]

        trimmed_messages = system_messages + other_messages
        self.messages = trimmed_messages
    
    def bot(self,message):
 
        self.messages.append({"role": "user", "content": message})
        llm_response = self.llm.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=self.messages,
            stream=True
        )

        response = ""
        for chunk in llm_response:
            if chunk.choices[0].delta.content:
                
                print(chunk.choices[0].delta.content, end="", flush=True)                
                response += chunk.choices[0].delta.content
                time.sleep(0.05)

            if chunk.usage:
                self.total_tokens += chunk.usage.total_tokens
                self.input_tokens += chunk.usage.prompt_tokens
                self.output_tokens += chunk.usage.completion_tokens
        
        self.messages.append({"role": "assistant", "content": response})

        self.trim_messages()

        return response
        # response_content = llm_response.choices[0].message.content
        
        # self.messages.append({"role": "assistant", "content": response_content})
        
        # self.total_tokens += llm_response.usage.total_tokens
        # self.input_tokens += llm_response.usage.prompt_tokens
        # self.output_tokens += llm_response.usage.completion_tokens
        
        # return response_content

    def chat(self):

        #tokenizer from tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        # encoding = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
        
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

            if self.total_tokens > 2000:
                print("Total token usage exceeds the limit of 100. Please start a new session.")
                break
            
            print(f"User input tokens: {token_count}")
            
            self.bot(user_input)
            print()

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
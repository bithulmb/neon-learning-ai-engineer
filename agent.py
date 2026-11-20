
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage,AIMessage
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()



def get_weather(location: str) -> dict:
    """
    Get the current weather for a given location
    args:
        location (str): The location to get the weather for
        returns:
            dict: A dictionary containing the weather information
    """
    # For demonstration purposes, we'll return a mock weather response.
    # In a real implementation, you would call a weather API here.
    return {
        "location": location,
        "temperature": "20°C",
        "condition": "Sunny"
    }

search_tool = TavilySearch(max_results=1)

class Agent:
    def __init__(self):
        self.llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.5
        )

        self.agent = create_agent(
            model=self.llm,
            tools=[get_weather, search_tool],
        )

        self.messages = [
            SystemMessage(content="You are a helpful assistant. You can use the provided tools if needed"),
        ]

    def chat_loop(self):
        print("Welcome to the ChatGroq agent! Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print("Exiting the chat. Goodbye!")
                break

            self.messages.append(HumanMessage(content=user_input))
            response = self.agent.invoke({"messages": self.messages})
            self.messages.append(response["messages"][-1])  # Append the AI's response to the messages
            print(f"AI: {response['messages'][-1].content}")
    
   


def main():
    agent = Agent()
    agent.chat_loop()

if __name__ == "__main__":
    main()
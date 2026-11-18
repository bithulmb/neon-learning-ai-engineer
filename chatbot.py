
def bot(message):
    answer = "bot: how can I help you?"
    return answer

def main():
    user_input = input("Ask something: ")
    bot_answer = bot(user_input)
    print(bot_answer)

if __name__ == "__main__":
    main()
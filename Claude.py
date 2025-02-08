import anthropic
import os

from anthropic.types import TextBlock
from dotenv import load_dotenv

#THIS IS THE API KEY IT SHOULD NOT BE PUSHED UNDER ANY CIRCUMSTANCES YOU SHOULD USE ENVIRONMENT VARIABLES INSTEAD!
load_dotenv()  # This will load the environment variables from the .env file
client = anthropic.Anthropic(api_key=os.getenv("claude_key"))

class Claude:
    #Temperature is set to 0 for deterministic results, set to higher value for non-deterministic
    def __init__(self, model: str="claude-3-5-sonnet-20240620", max_tokens: int =1024, temperature: int =0.7, source_text: str=''):
        # Check types of the parameters
        if not isinstance(model, str):
            raise TypeError("The 'model' parameter must be a string.")

        if not isinstance(max_tokens, int):
            raise TypeError("The 'max_tokens' parameter must be an integer.")

        if not isinstance(temperature, float):
            raise TypeError("The 'temperature' parameter must be an integer.")

        # Initialize attributes
        self.source_text = source_text

        self.model = model

        self.temperature = temperature
        self.max_tokens = max_tokens

    def __send_message(self, question: str) -> str:
        system = (f"You are an AI assistant that answers questions based on the provided source text. "
                  f"Your responses should be accurate, concise, and directly relevant to the question. "
                  f"Use only the information in the source text to answer.\n\n"
                  f"Source Text:\n{self.source_text}\n\n"
                  f"Question:\n{question}\n\n"
                  f"Answer:")
        message: TextBlock = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[
                {
                    'role' : 'user', 'content' : [{"type": "text", "text":question}]
                }
            ]
        )
        print(f"message: {message.content[0].text}")
        return message.content[0].text
    def send_message_debug(self, question: str) -> str:
        print("Source Text:\n" + self.source_text + '\n' + '-'*50)
        return 'This is the answer to the question'

    def set_source_text(self, new_source_text:str) -> None:
        self.source_text = new_source_text

    def invoke(self, user_prompt: str) -> str:
        return self.__send_message(question=user_prompt)
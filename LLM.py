from importlib.util import source_hash

import anthropic

#THIS IS THE API KEY IT SHOULD NOT BE PUSHED UNDER ANY CIRCUMSTANCES YOU SHOULD USE ENVIRONMENT VARIABLES INSTEAD!
client = anthropic.Anthropic()#api_key='***')

class Claude:
    #Temperature is set to 0 for deterministic results, set to higher value for non-deterministic
    def __init__(self, model: str="claude-3-5-sonnet-20240620", max_tokens: int =1024, temperature: int =0, source_text: str=''):
        # Check types of the parameters
        if not isinstance(model, str):
            raise TypeError("The 'model' parameter must be a string.")

        if not isinstance(max_tokens, int):
            raise TypeError("The 'max_tokens' parameter must be an integer.")

        if not isinstance(temperature, int):
            raise TypeError("The 'temperature' parameter must be an integer.")

        if not isinstance(source_text, str):
            raise TypeError("The 'source_text' parameter must be a string.")
        if source_text == '':
            raise ValueError("failed to initialize source text\n Source Text can not be empty string!")

        # Initialize attributes
        self.source_text = source_text

        self.model = model

        self.temperature = temperature
        self.max_tokens = max_tokens

    def send_message(self, question: str):
        system = (f"You are an AI assistant that answers questions based on the provided source text. "
                  f"Your responses should be accurate, concise, and directly relevant to the question. "
                  f"Use only the information in the source text to answer.\n\n"
                  f"Source Text:\n{self.source_text}\n\n"
                  f"Question:\n{question}\n\n"
                  f"Answer:")
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[
                {
                    'role' : 'user', 'content' : question
                }
            ]
        )
        return message.content
    def send_message_debug(self, question: str):
        print("Source Text:\n" + self.source_text + '\n' + '-'*50)
        return 'This is the answer to the question'


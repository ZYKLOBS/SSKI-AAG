import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class OpenAIWrapper:
    def __init__(self, model: str = "gpt-4o", max_tokens: int = 1024, temperature: float = 0.7, source_text: str = ''):
        if not isinstance(model, str):
            raise TypeError("The 'model' parameter must be a string.")
        if not isinstance(max_tokens, int):
            raise TypeError("The 'max_tokens' parameter must be an integer.")
        if not isinstance(temperature, float):
            raise TypeError("The 'temperature' parameter must be a float.")

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.source_text = source_text
        self.client = OpenAI(api_key="NO_VALUE")  # To be set in invoke/refine

    def __build_system_prompt(self, question: str, prompt_template: str) -> str:
        return (
            f"You are an AI assistant that answers questions based on the provided source text and a prompt_template given by the user. "
            f"If the prompt_template is 'Default' you can ignore it. "
            f"Your responses should be accurate, concise, and directly relevant to the question. "
            f"Use only the information in the source text to answer.\n\n"
            f"Source Text:\n{self.source_text}\n\n"
            f"Prompt Template:\n{prompt_template}\n\n"
            f"Question:\n{question}\n\n"
            f"Answer:"
        )

    def __build_refine_system_prompt(self, question: str, prompt_template: str, refine_prompt: str, previous_answer: str) -> str:
        return (
            f"You are an AI assistant that answers questions based on the provided source text and a prompt_template given by the user. "
            f"If the prompt_template is 'Default' you can ignore it. "
            f"Your responses should be accurate, concise, and directly relevant to the question. "
            f"You have been instructed to refine your previous answer according to the refine instructions, please do so. "
            f"Use only the information in the source text to answer.\n\n"
            f"Source Text:\n{self.source_text}\n\n"
            f"Prompt Template:\n{prompt_template}\n\n"
            f"Refine Instructions:\n{refine_prompt}\n\n"
            f"Question:\n{question}\n\n"
            f"Previous Answer:\n{previous_answer}\n\n"
            f"Answer:"
        )

    def __send_message(self, question: str, prompt_template: str) -> str:
        system_prompt = self.__build_system_prompt(question, prompt_template)
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
        )
        answer = response.choices[0].message.content
        print(f"message: {answer}")
        return answer

    def __refine(self, question: str, refine_prompt: str, previous_answer: str, prompt_template: str) -> str:
        system_prompt = self.__build_refine_system_prompt(question, prompt_template, refine_prompt, previous_answer)
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
        )
        answer = response.choices[0].message.content
        print(f"message: {answer}")
        return answer

    def send_message_debug(self, question: str) -> str:
        print("Source Text:\n" + self.source_text + '\n' + '-' * 50)
        return 'This is the answer to the question'

    def set_source_text(self, new_source_text: str) -> None:
        self.source_text = new_source_text

    def invoke(self, user_prompt: str, prompt_template: str, api_key: str) -> str:
        self.client = OpenAI(api_key=api_key)
        return self.__send_message(question=user_prompt, prompt_template=prompt_template)

    def refine(self, user_prompt: str, prompt_template: str, refine_prompt: str, previous_answer: str, api_key: str) -> str:
        self.client = OpenAI(api_key=api_key)
        return self.__refine(question=user_prompt, refine_prompt=refine_prompt, previous_answer=previous_answer, prompt_template=prompt_template)

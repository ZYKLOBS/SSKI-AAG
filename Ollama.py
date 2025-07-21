from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

class OllamaWrapper:
    def __init__(
        self,
        model: str = "llama3.1:latest", #Select model here yourself in code !
        max_tokens: int = 2048, #default https://github.com/ollama/ollama/issues/3643
        temperature: float = 0.8, #default https://github.com/ollama/ollama/issues/6410
        source_text: str = '',
    ):
        if not isinstance(model, str):
            raise TypeError("The 'model' parameter must be a string.")
        if not isinstance(max_tokens, int):
            raise TypeError("The 'max_tokens' parameter must be an integer.")
        if not isinstance(temperature, float):
            raise TypeError("The 'temperature' parameter must be a float.")

        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.source_text = source_text
        self.client = OllamaLLM(model=self.model_name)

        self.template_str = (
            "You are an AI assistant that answers questions based on the provided source text and a prompt_template given by the user. "
            "If the prompt_template is 'Default' you can ignore it. "
            "Your responses should be accurate, concise, and directly relevant to the question. "
            "Use only the information in the source text to answer.\n\n"
            "Source Text:\n{source_text}\n\n"
            "Prompt Template:\n{prompt_template}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        )

        self.refine_template_str = (
            "You are an AI assistant that answers questions based on the provided source text and a prompt_template given by the user. "
            "If the prompt_template is 'Default' you can ignore it. "
            "Your responses should be accurate, concise, and directly relevant to the question. "
            "You have been instructed to refine your previous answer according to the refine instructions, please do so. "
            "Use only the information in the source text to answer.\n\n"
            "Source Text:\n{source_text}\n\n"
            "Prompt Template:\n{prompt_template}\n\n"
            "Refine Instructions:\n{refine_prompt}\n\n"
            "Question:\n{question}\n\n"
            "Previous Answer:\n{previous_answer}\n\n"
            "Answer:"
        )

    def set_source_text(self, new_source_text: str) -> None:
        self.source_text = new_source_text

    def __send_message(self, question: str, prompt_template: str) -> str:
        if not self.source_text:
            raise ValueError("Source text cannot be empty.")

        prompt_template_obj = ChatPromptTemplate.from_template(self.template_str)
        prompt = prompt_template_obj.format(
            source_text=self.source_text,
            prompt_template=prompt_template,
            question=question,
        )

        answer = self.client.invoke(prompt)
        print(f"prompt: {prompt}")
        print(f"answer: {answer}")
        return answer

    def __refine(self, question: str, refine_prompt: str, previous_answer: str, prompt_template: str) -> str:
        if not self.source_text:
            raise ValueError("Source text cannot be empty.")

        prompt_template_obj = ChatPromptTemplate.from_template(self.refine_template_str)
        prompt = prompt_template_obj.format(
            source_text=self.source_text,
            prompt_template=prompt_template,
            refine_prompt=refine_prompt,
            question=question,
            previous_answer=previous_answer,
        )

        answer = self.client.invoke(prompt)
        print(f"refine prompt: {prompt}")
        print(f"refined answer: {answer}")
        return answer

    def send_message_debug(self, question: str) -> str:
        print("Source Text:\n" + self.source_text + "\n" + "-" * 50)
        return "This is the answer to the question"

    def invoke(self, user_prompt: str, prompt_template: str, api_key: str = None) -> str:
        # OllamaLLM might not require api_key, but keeping signature consistent
        return self.__send_message(question=user_prompt, prompt_template=prompt_template)

    def refine(
        self, user_prompt: str, prompt_template: str, refine_prompt: str, previous_answer: str, api_key: str = None
    ) -> str:
        return self.__refine(
            question=user_prompt,
            refine_prompt=refine_prompt,
            previous_answer=previous_answer,
            prompt_template=prompt_template,
        )

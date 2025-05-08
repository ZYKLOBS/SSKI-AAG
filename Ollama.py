from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

class Ollama:
    def __init__(self, model:str = "gemma3", source_text=''):
        self.model = OllamaLLM(model=model)
        self.source_text = source_text

    template = ("You are an AI assistant that answers questions based on the provided source text. "
                  "Your responses should be accurate, concise, and directly relevant to the question."
                  "Use only the information in the source text to answer.\n\n"
                  "Source Text:\n{source_text}\n\n"
                  "Question:\n{question}\n\n"
                  "Answer:")
    def set_source_text(self, new_source_text:str) -> None:
        self.source_text = new_source_text

    def invoke(self, user_prompt: str) -> str:
        if self.source_text == '':
            raise ValueError("Base Text may not be empty")
        prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_template(self.template)
        prompt: str = prompt_template.format(source_text=self.source_text, question=user_prompt)
        answer: str = self.model.invoke(prompt)
        print(f'prompt: {prompt}')
        return answer

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn when the script is executed
    uvicorn.run("main_ollama:app", host="127.0.0.1", port=8000, reload=True)

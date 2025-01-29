from LLM import Claude
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (e.g., index.html)
app.mount("/.", StaticFiles(directory="/."), name="static")

llm = Claude(source_text="""test """)

class TextRequest(BaseModel):
    text: str

@app.get("/")
async def get_index():
    return FileResponse("./index.html")

@app.post("/process")
async def process_text(request: TextRequest):
    response = llm.send_message_debug(request.text)  # Assuming `generate_response` is a method of `Claude`
    print(f"Response from LLM: {response}")  # Log the response for debugging
    return {"response": response}


if __name__ == "__main__":
    # Run the FastAPI app with uvicorn when the script is executed
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
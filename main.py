from Claude import Claude
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

app = FastAPI()
#TODO: Fragenkatalog aus Excel datei laden, extra prompt for template string -> specify things for questions -> hint fuer fragen um antwort in richtige richtung,
#TODO: Char limit -> erstmal char counter fuer convenience/debug
#TODO: Antwort neu generieren, Fragen loeschen, Fragen speichern -> EXCEL??? -> ja
#TODO: API Key und Model auswaehlen oben

# CORS middleware allows the frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (e.g., index.html)
app.mount("/static", StaticFiles(directory="."), name="static")

# Initialize the Claude LLM with default source text
llm = Claude()
class TextRequest(BaseModel):
    text: str

class SourceTextRequest(BaseModel):
    new_text: str  # New source text to set

# Serve the index.html page when the user accesses the root path
@app.get("/")
async def get_index():
    return FileResponse("./index.html")

# Endpoint to process the text input and return a response
@app.post("/process")
async def process_text(request: TextRequest):
    print(f"Received request: {request.text}")  # Log question
    print(f"Using source text: {llm.source_text}")  # Log the source text being used
    response = llm.invoke(request.text)  # Get the response from LLM
    print(f'response: {response}')
    return {"response": response}


# Endpoint to update the source text dynamically
@app.post("/set_source_text")
async def set_source_text(request: SourceTextRequest):
    print(f"Setting new source text: {request.new_text}")
    llm.set_source_text(request.new_text)  # Set the new source text for the model
    return {"message": "Source text updated successfully."}


if __name__ == "__main__":
    # Run the FastAPI app with uvicorn when the script is executed
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

from LLM import Ollama
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
app.mount("/.", StaticFiles(directory="/.", html=True), name="static")

llm = Ollama(source_text="""The Embodiment of Scarlet Devil's gameplay holds various fundamental similarities to that of the PC-98 titles and "Seihou", but differs in the introduction of Spell Cards. The pacing has also been changed significantly, but retains some of the PC-98 titles' fast pacing, giving the Embodiment of Scarlet Devil a reputation for having a heightened difficulty compared to other Windows titles. These aspects were, for the post part, removed in "Perfect Cherry Blossom" and "Imperishable Night". A change retained from the PC-98 titles is the "Item Get" feature, which allows the player to collect all items on the screen by moving to the top of the screen, if at full power. This mechanic was retained in all later games.

the Embodiment of Scarlet Devil features two playable characters to choose from with two equipment types each; Reimu Hakurei can cover a wide area of the screen with weaker attacks, while Marisa Kirisame relies on speed and power to make up for a thinner attack spread. Each character and type has its own spell card as well.

Among the Embodiment of Scarlet Devil's similarities to the PC-98 titles is the game containing six stages in total; however, the player is prevented from continuing to Stage 6 from Stage 5 if playing on Easy difficulty. """)

class TextRequest(BaseModel):
    text: str

@app.get("/")
async def get_index():
    return FileResponse("./index.html")

@app.post("/process")
async def process_text(request: TextRequest):
    print(f"Received request: {request.text}")
    response = llm.invoke(request.text)
    print(f"Response from LLM: {response}")  # Log the response for debugging
    return {"response": response}


if __name__ == "__main__":
    # Run the FastAPI app with uvicorn when the script is executed
    uvicorn.run("main_ollama:app", host="127.0.0.1", port=8000, reload=True)
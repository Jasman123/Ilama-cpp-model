# main.py — Level 2 entry point
#
# WHY is main.py so short now?
#   Because all the real work is in app/.
#   main.py's only job is to assemble the pieces.

import os
from fastapi import FastAPI
from dotenv import load_dotenv
from app.routes import router  # Import the router we defined in routes.py

load_dotenv()  # Load .env before anything reads environment variables

app = FastAPI(
    title="Llama API — Level 2",
    description="Intermediate: modules, streaming, chat",
    version="2.0.0",
)

# include_router attaches all the routes defined in routes.py to our app.
# prefix="/api/v1" means all routes get that prepended:
#   /health → /api/v1/health
#   /generate → /api/v1/generate
app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router as api_router
import os
from dotenv import load_dotenv
import logging

load_dotenv()

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Firebase only if credentials are available
firebase_initialized = False
try:
    if os.getenv('FIREBASE_CREDENTIALS_PATH'):
        from firebase_admin import credentials, initialize_app
        cred = credentials.Certificate(os.getenv('FIREBASE_CREDENTIALS_PATH'))
        initialize_app(cred)
        firebase_initialized = True
        logging.info("Firebase initialized successfully")
    else:
        logging.warning("FIREBASE_CREDENTIALS_PATH not found. Using in-memory storage instead.")
except Exception as e:
    logging.error(f"Failed to initialize Firebase: {str(e)}")

# Add global flag to app state that can be checked by routes
app.state.firebase_initialized = firebase_initialized

app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
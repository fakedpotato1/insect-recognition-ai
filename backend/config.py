import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    USE_MOCK_MODEL = (
        os.getenv("USE_MOCK_MODEL", "true")
        .lower()
        == "true"
    )

    PORT = int(
        os.getenv("PORT", 5000)
    )
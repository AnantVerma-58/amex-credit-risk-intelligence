
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, text

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

url = URL.create(
    drivername="mysql+pymysql",
    username=os.environ["DB_USER"],
    password=os.environ["DB_PASSWORD"],
    host=os.getenv("DB_HOST", "127.0.0.1"),
    port=int(os.getenv("DB_PORT", "3306")),
    database=os.environ["DB_NAME"],
)

engine = create_engine(
    url,
    pool_pre_ping=True,
    connect_args={"connect_timeout": 10},
)

with engine.connect() as conn:
    result = conn.execute(
        text("SELECT DATABASE(), VERSION(), CURRENT_USER()")
    ).one()

    print("Database:", result[0])
    print("Server version:", result[1])
    print("Authenticated user:", result[2])

engine.dispose()
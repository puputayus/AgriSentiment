import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    db_host = os.getenv("DB_HOST")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_port = os.getenv("DB_PORT")

    if not all([db_host, db_name, db_user, db_password, db_port]):
        raise Exception("Database environment variables belum lengkap di file .env")

    return psycopg2.connect(
        host=db_host,
        database=db_name,
        user=db_user,
        password=db_password,
        port=db_port
    )


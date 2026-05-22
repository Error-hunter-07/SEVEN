import psycopg2 
from psycopg2 import OperationalError
import os
from dotenv import load_dotenv

load_dotenv()

class DB:

    def __init__(self):
        self.connection_string = (
            "postgresql://" + os.getenv("DB_USER") + ":" + os.getenv("DB_PASSWORD") + "@localhost:5432/" + os.getenv("DB_NAME")
        )

    def connect(self):
        try:
            conn = psycopg2.connect(self.connection_string)
            print("Connection to database successful")
            return conn
        except OperationalError as e:
            print(f"The error '{e}' occurred, Could not connect to Database")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None


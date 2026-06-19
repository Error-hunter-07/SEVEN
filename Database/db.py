import psycopg2
from psycopg2 import pool, OperationalError
import os
from dotenv import load_dotenv

load_dotenv()

class DB:

    def __init__(self, minconn: int = 2, maxconn: int = 10):
        self.connection_string = (
            "postgresql://"
            + os.getenv("DB_USER") + ":"
            + os.getenv("DB_PASSWORD")
            + "@localhost:5432/"
            + os.getenv("DB_NAME")
        )
        self._pool: psycopg2.pool.ThreadedConnectionPool | None = None
        self._minconn = minconn
        self._maxconn = maxconn
        self._init_pool()

    def _init_pool(self):
        try:
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                self._minconn,
                self._maxconn,
                self.connection_string
            )
            print("[DB] Connection pool initialised successfully")
        except OperationalError as e:
            print(f"[DB] Could not initialise connection pool: {e}")
        except Exception as e:
            print(f"[DB] Unexpected error during pool init: {e}")

    def get_connection(self):
        if self._pool is None:
            print("[DB] Pool is not available.")
            return None
        try:
            return self._pool.getconn()
        except Exception as e:
            print(f"[DB] Failed to get connection from pool: {e}")
            return None

    def put_connection(self, conn):
        if self._pool and conn:
            self._pool.putconn(conn)

    def close_all(self):
        if self._pool:
            self._pool.closeall()
            print("[DB] All pooled connections closed")

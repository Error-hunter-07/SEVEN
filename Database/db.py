import psycopg2
from psycopg2 import pool, OperationalError
import os
from dotenv import load_dotenv
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)

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
            log.info("Connection pool initialised successfully")
        except OperationalError as e:
            log.error("Could not initialise connection pool: %s", e, exc_info=True)
        except Exception as e:
            log.error("Unexpected error during pool init: %s", e, exc_info=True)

    def get_connection(self):
        if self._pool is None:
            log.warning("Pool is not available.")
            return None
        try:
            return self._pool.getconn()
        except Exception as e:
            log.error("Failed to get connection from pool: %s", e, exc_info=True)
            return None

    def put_connection(self, conn):
        if self._pool and conn:
            self._pool.putconn(conn)

    def close_all(self):
        if self._pool:
            self._pool.closeall()
            log.info("All pooled connections closed")

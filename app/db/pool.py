"""
Shared PostgreSQL connection pool.
Initialized once at first use; reused for every subsequent DB call.
Azure Flexible Server requires sslmode=require.
"""
import psycopg2.pool

from app.config import config

_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        if not all([
            config.PGSQL_ENDPOINT,
            config.PGSQL_DB_NAME,
            config.PGSQL_ADMIN_USERNAME,
            config.PGSQL_ADMIN_PASSWORD,
        ]):
            raise ValueError(
                "Set PGSQL_ENDPOINT, PGSQL_DB_NAME, PGSQL_ADMIN_USERNAME, "
                "PGSQL_ADMIN_PASSWORD in .env"
            )
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            host=config.PGSQL_ENDPOINT,
            dbname=config.PGSQL_DB_NAME,
            user=config.PGSQL_ADMIN_USERNAME,
            password=config.PGSQL_ADMIN_PASSWORD,
            port=5432,
            sslmode="require",
        )
    return _pool

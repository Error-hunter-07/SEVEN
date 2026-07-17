import json
from multiprocessing.dummy import connection
from Database import db
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)

conn_pool = db.DB()


# This is the SQL schema for the working_memory table in PostgreSQL. It defines the structure of the table, including the columns, their data types, and constraints. The table is designed to store information related to working memory, such as session ID, memory type, key-value pairs, priority, relevance, timestamps, source, tags, access count, and active status.
# CREATE TABLE IF NOT EXISTS public.working_memory
# (
#     id uuid NOT NULL,
#     session_id uuid,
#     memory_type character varying(50) COLLATE pg_catalog."default",
#     key text COLLATE pg_catalog."default",
#     value jsonb,
#     priority double precision DEFAULT 0.5,
#     relevance double precision DEFAULT 0.5,
#     created_at timestamp without time zone,
#     updated_at timestamp without time zone,
#     expires_at timestamp without time zone,
#     source text COLLATE pg_catalog."default",
#     tags text[] COLLATE pg_catalog."default",
#     access_count integer DEFAULT 0,
#     last_accessed timestamp without time zone,
#     active boolean DEFAULT true,
#     CONSTRAINT working_memory_pkey PRIMARY KEY (id)
# )


def insert_working_memory(session_id, memory_type, key, value, priority=0.5,relevance=0.5, source=None, tags=None):
    connection = conn_pool.get_connection()
    if connection is None:
        log.warning("insert_working_memory: Failed to get connection from pool.")
        return None

    try:
        with connection.cursor() as cursor:
            insert_query = """
                INSERT INTO public.working_memory (
                    id, session_id, memory_type, key, value,
                    priority, relevance,
                    created_at, updated_at, expires_at,
                    source, tags
                )
                VALUES (
                    gen_random_uuid(),
                    %s, %s, %s, %s::jsonb,
                    %s, %s,
                    NOW(), NOW(), %s,
                    %s, %s
                )
                RETURNING id;
            """
            cursor.execute(
                insert_query,
                (
                    session_id,
                    memory_type,
                    key,
                    json.dumps(value),
                    priority,
                    relevance,
                    None,
                    source,
                    tags
                )
            )
            row = cursor.fetchone()
            new_id = row[0] if row else None
            connection.commit()
            return new_id
    except Exception as e:
        log.error("insert_working_memory error: %s: %s", type(e).__name__, e, exc_info=True)
        connection.rollback()
        return None
    finally:
        conn_pool.put_connection(connection)


def get_working_memory(session_id):
    connection = conn_pool.get_connection()
    if connection is None:
        log.warning("get_working_memory: Failed to get connection from pool.")
        return None

    try:
        with connection.cursor() as cursor:
            select_query = """
                SELECT id, memory_type, key, value, priority, relevance,
                       created_at, updated_at, expires_at, source, tags
                FROM public.working_memory
                WHERE session_id = %s AND active = true
                ORDER BY created_at DESC
                LIMIT 1;
            """
            cursor.execute(select_query, (session_id,))
            results = cursor.fetchall()
            return results
    except Exception as e:
        log.error("get_working_memory error: %s", e, exc_info=True)
        return None
    finally:
        conn_pool.put_connection(connection)


def update_working_memory(memory_id, key=None, value=None, priority=None, relevance=None, expires_at=None, source=None,tags=None):
    if not memory_id:
        log.warning("update_working_memory: called with no memory_id — refusing to run a no-op update.")
        return False

    connection = conn_pool.get_connection()
    if connection is None:
        log.warning("update_working_memory: Failed to get connection from pool.")
        return False

    try:
        with connection.cursor() as cursor:
            update_fields = []
            update_values = []

            if key is not None:
                update_fields.append("key = %s")
                update_values.append(key)
            if value is not None:
                update_fields.append("value = %s::jsonb")
                update_values.append(json.dumps(value))
            if priority is not None:
                update_fields.append("priority = %s")
                update_values.append(priority)
            if relevance is not None:
                update_fields.append("relevance = %s")
                update_values.append(relevance)
            if expires_at is not None:
                update_fields.append("expires_at = %s")
                update_values.append(expires_at)
            if source is not None:
                update_fields.append("source = %s")
                update_values.append(source)
            if tags is not None:
                update_fields.append("tags = %s")
                update_values.append(tags)

            if not update_fields:
                log.warning("update_working_memory: No fields to update.")
                return False

            update_query = f"""
                UPDATE public.working_memory
                SET {', '.join(update_fields)}, updated_at = NOW()
                WHERE id = %s::uuid;
            """
            cursor.execute(update_query, (*update_values, memory_id))
            if cursor.rowcount == 0:
                log.warning("update_working_memory: no row matched id=%s — nothing was updated.", memory_id)
                connection.rollback()
                return False
            connection.commit()
            return True
    except Exception as e:
        log.error("update_working_memory error: %s", e, exc_info=True)
        connection.rollback()
        return False
    finally:
        conn_pool.put_connection(connection)


def get_all_current_session_working_memory(session_id):
    connection = conn_pool.get_connection()
    if connection is None:
        log.warning("get_all_current_session_working_memory: Failed to get connection from pool.")
        return None

    try:
        with connection.cursor() as cursor:
            select_query = """
                SELECT id, memory_type, key, value, priority, relevance,
                       created_at, updated_at, expires_at, source, tags
                FROM public.working_memory
                WHERE session_id = %s AND active = true
                ORDER BY created_at DESC;
            """
            cursor.execute(select_query, (session_id,))
            results = cursor.fetchall()
            return results
    except Exception as e:
        log.error("get_all_current_session_working_memory error: %s", e, exc_info=True)
        return None
    finally:
        conn_pool.put_connection(connection)

# def delete_working_memory(memory_id):
#     connection = conn.connect()
#     if connection is None:
#         print("Failed to connect to the database.")
#         return False

#     try:
#         with connection.cursor() as cursor:
#             delete_query = """
#                 UPDATE public.working_memory
#                 SET active = false, updated_at = NOW()
#                 WHERE id = %s::uuid;
#             """
#             cursor.execute(delete_query, (memory_id,))
#             connection.commit()
#             return True
#     except Exception as e:
#         print(f"An error occurred while deleting working memory: {e}")
#         connection.rollback()
#         return False
#     finally:
#         connection.close()
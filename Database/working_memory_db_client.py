from Database import db


conn = db.DB()

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


-- Enable the pgvector extension
create extension if not exists vector;

-- Create the documentation chunks table
create table site_pages (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    title varchar not null,
    summary varchar not null,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    embedding vector(1024),  -- Changed to 1024 dimensions for mxbai-embed-large
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,

    -- Unique constraint remains the same
    unique(url, chunk_number)
);

-- Create index with updated vector dimensions
create index on site_pages using ivfflat (embedding vector_cosine_ops)
with (lists = 100);  -- Adjusted for 1024-dimensional vectors

-- Metadata index remains the same
create index idx_site_pages_metadata on site_pages using gin (metadata);

-- Updated function for 1024-dimensional vectors
create or replace function match_site_pages (
  query_embedding vector(1024),  -- Changed dimension
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb
) returns table (
  id bigint,
  url varchar,
  chunk_number integer,
  title varchar,
  summary varchar,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    url,
    chunk_number,
    title,
    summary,
    content,
    metadata,
    1 - (site_pages.embedding <=> query_embedding) as similarity
  from site_pages
  where metadata @> filter
  order by site_pages.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Security policies remain the same
alter table site_pages enable row level security;

create policy "Allow public read access"
  on site_pages
  for select
  to public
  using (true);
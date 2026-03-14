create extension if not exists pgcrypto;

create table if not exists generations (
  id uuid primary key default gen_random_uuid(),
  client_run_id text not null unique,
  user_prompt text not null,
  full_prompt text not null,
  modal_path text not null,
  content_type text not null default 'image/png',
  width integer not null,
  height integer not null,
  status text not null default 'generated',
  created_at timestamptz not null default now()
);

create table if not exists artworks (
  id uuid primary key default gen_random_uuid(),
  generation_id uuid not null unique references generations(id) on delete cascade,
  slug text not null unique,
  title text not null,
  attribution text not null,
  created_at timestamptz not null default now(),
  published_at timestamptz not null default now()
);

create index if not exists artworks_published_at_idx on artworks (published_at desc);

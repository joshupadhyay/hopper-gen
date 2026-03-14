import postgres from "postgres";
import slugify from "slugify";

import { appConfig, requireConfigValue } from "@/lib/config";
import type {
  ArtworkRecord,
  GenerationRecord,
  PublishedArtworkSummary,
} from "@/lib/types";

let sql: postgres.Sql | null = null;

let schemaPromise: Promise<void> | null = null;

function getSql() {
  if (!sql) {
    sql = postgres(requireConfigValue(appConfig.databaseUrl, "DATABASE_URL"), {
      prepare: false,
      max: 5,
    });
  }

  return sql;
}

async function ensureSchema() {
  if (!schemaPromise) {
    schemaPromise = (async () => {
      const db = getSql();

      await db`
        create extension if not exists pgcrypto;
      `;

      await db`
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
      `;

      await db`
        create table if not exists artworks (
          id uuid primary key default gen_random_uuid(),
          generation_id uuid not null unique references generations(id) on delete cascade,
          slug text not null unique,
          title text not null,
          attribution text not null,
          created_at timestamptz not null default now(),
          published_at timestamptz not null default now()
        );
      `;

      await db`
        create index if not exists artworks_published_at_idx on artworks (published_at desc);
      `;

      await db`
        alter table generations
        add column if not exists client_run_id text;
      `;

      await db`
        update generations
        set client_run_id = id::text
        where client_run_id is null;
      `;

      await db`
        alter table generations
        alter column client_run_id set not null;
      `;

      await db`
        create unique index if not exists generations_client_run_id_idx
        on generations (client_run_id);
      `;
    })();
  }

  await schemaPromise;
}

function mapGeneration(row: any): GenerationRecord {
  return {
    id: row.id,
    clientRunId: row.client_run_id,
    userPrompt: row.user_prompt,
    fullPrompt: row.full_prompt,
    modalPath: row.modal_path,
    contentType: row.content_type,
    width: row.width,
    height: row.height,
    status: row.status,
    createdAt: row.created_at,
  };
}

function mapArtwork(row: any): ArtworkRecord {
  return {
    id: row.id,
    generationId: row.generation_id,
    slug: row.slug,
    title: row.title,
    attribution: row.attribution,
    createdAt: row.created_at,
    publishedAt: row.published_at,
    modalPath: row.modal_path,
    contentType: row.content_type,
    userPrompt: row.user_prompt,
  };
}

export async function createGeneration(input: {
  clientRunId: string;
  userPrompt: string;
  fullPrompt: string;
  modalPath: string;
  contentType: string;
  width: number;
  height: number;
}) {
  await ensureSchema();
  const [row] = await getSql()`
    insert into generations (
      client_run_id,
      user_prompt,
      full_prompt,
      modal_path,
      content_type,
      width,
      height
    ) values (
      ${input.clientRunId},
      ${input.userPrompt},
      ${input.fullPrompt},
      ${input.modalPath},
      ${input.contentType},
      ${input.width},
      ${input.height}
    )
    returning *;
  `;

  return mapGeneration(row);
}

export async function getGeneration(id: string) {
  if (!appConfig.databaseUrl) {
    return null;
  }

  await ensureSchema();
  const [row] = await getSql()`
    select *
    from generations
    where id = ${id}
    limit 1;
  `;

  return row ? mapGeneration(row) : null;
}

export async function publishArtwork(input: {
  generationId: string;
  title: string;
  attribution: string;
}) {
  await ensureSchema();
  const baseSlug = slugify(input.title, { lower: true, strict: true }) || "untitled";

  const [generationRow] = await getSql()`
    select *
    from generations
    where id = ${input.generationId}
    limit 1;
  `;

  if (!generationRow) {
    throw new Error("Generation not found");
  }

  const [existing] = await getSql()`
    select a.*, g.modal_path, g.content_type, g.user_prompt
    from artworks a
    join generations g on g.id = a.generation_id
    where a.generation_id = ${input.generationId}
    limit 1;
  `;

  if (existing) {
    return mapArtwork(existing);
  }

  let slug = baseSlug;
  let suffix = 1;

  while (true) {
    const [taken] = await getSql()`
      select slug
      from artworks
      where slug = ${slug}
      limit 1;
    `;

    if (!taken) {
      break;
    }

    suffix += 1;
    slug = `${baseSlug}-${suffix}`;
  }

  const [row] = await getSql()`
    insert into artworks (
      generation_id,
      slug,
      title,
      attribution
    ) values (
      ${input.generationId},
      ${slug},
      ${input.title.trim()},
      ${input.attribution.trim()}
    )
    returning *;
  `;

  const [joined] = await getSql()`
    select a.*, g.modal_path, g.content_type, g.user_prompt
    from artworks a
    join generations g on g.id = a.generation_id
    where a.id = ${row.id}
    limit 1;
  `;

  return mapArtwork(joined);
}

export async function listPublishedArtworks(limit = 12) {
  if (!appConfig.databaseUrl) {
    return [];
  }

  await ensureSchema();
  const rows = await getSql()`
    select slug, title, attribution, published_at
    from artworks
    order by published_at desc
    limit ${limit};
  `;

  return rows.map(
    (row): PublishedArtworkSummary => ({
      slug: row.slug,
      title: row.title,
      attribution: row.attribution,
      publishedAt: row.published_at,
    }),
  );
}

export async function getArtworkBySlug(slug: string) {
  if (!appConfig.databaseUrl) {
    return null;
  }

  await ensureSchema();
  const [row] = await getSql()`
    select a.*, g.modal_path, g.content_type, g.user_prompt
    from artworks a
    join generations g on g.id = a.generation_id
    where a.slug = ${slug}
    limit 1;
  `;

  return row ? mapArtwork(row) : null;
}

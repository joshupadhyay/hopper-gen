import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { appConfig } from "@/lib/config";
import { formatDisplayDate } from "@/lib/date";
import { getArtworkBySlug } from "@/lib/db";

export const dynamic = "force-dynamic";

type Props = {
  params: Promise<{ slug: string }>;
};

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const artwork = await getArtworkBySlug(slug);

  if (!artwork) {
    return {
      title: "Artwork not found",
    };
  }

  const imageUrl = `${appConfig.siteUrl}/artwork/${artwork.slug}/image`;

  return {
    title: `${artwork.title} | Hopper Studio`,
    description: `Published by ${artwork.attribution}.`,
    openGraph: {
      images: [imageUrl],
    },
  };
}

export default async function ArtworkPage({ params }: Props) {
  const { slug } = await params;
  const artwork = await getArtworkBySlug(slug);

  if (!artwork) {
    notFound();
  }

  return (
    <main className="detail-shell">
      <section className="art-stage">
        <div className="frame">
          <img
            alt={artwork.title}
            className="art-image"
            src={`/artwork/${artwork.slug}/image`}
          />
        </div>
        <div className="plaque">
          <strong>{artwork.title}</strong>
          <span>{artwork.attribution}</span>
          <span>{formatDisplayDate(artwork.publishedAt)}</span>
        </div>
      </section>

      <section className="detail-sidebar">
        <p className="eyebrow">Published Work</p>
        <h1>{artwork.title}</h1>
        <p className="lede">
          A public permalink backed by Modal-stored source imagery and Vercel
          metadata.
        </p>

        <div className="prompt-summary">
          <strong>Attribution</strong>
          <p>{artwork.attribution}</p>
        </div>

        <div className="prompt-summary">
          <strong>Prompt</strong>
          <p>{artwork.userPrompt}</p>
        </div>

        <div className="prompt-summary">
          <strong>Permalink</strong>
          <p>{`${appConfig.siteUrl}/artwork/${artwork.slug}`}</p>
        </div>
      </section>
    </main>
  );
}

import { HomepageClient } from "@/components/homepage-client";
import { listPublishedArtworks } from "@/lib/db";
import type { PublishedArtworkSummary } from "@/lib/types";

export const dynamic = "force-dynamic";

const STATIC_SAMPLES: PublishedArtworkSummary[] = [
  { slug: "pool-scene", title: "Sunny Day at the Pool", attribution: "hopper style", publishedAt: new Date() },
  { slug: "picnic-river", title: "Picnic on the River", attribution: "hopper style", publishedAt: new Date() },
  { slug: "skyline-hopper", title: "Central Park Skyline", attribution: "hopper style", publishedAt: new Date() },
  { slug: "west-side-highway", title: "West Side Highway", attribution: "hopper style", publishedAt: new Date() },
];

export default async function HomePage() {
  const dbArtworks = await listPublishedArtworks(10);
  const artworks = dbArtworks.length > 0 ? dbArtworks : STATIC_SAMPLES;

  return <HomepageClient artworks={artworks} />;
}

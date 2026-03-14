import { readFile } from "node:fs/promises";
import { join } from "node:path";

import { fetchModalFile } from "@/lib/modal";
import { getArtworkBySlug } from "@/lib/db";

const STATIC_SAMPLES: Record<string, string> = {
  "pool-scene": "hopper_lora_square_0.png",
  "picnic-river": "picnic_river_0.png",
  "skyline-hopper": "skyline_hopper.png",
  "west-side-highway": "westsidehighway.png",
};

type Props = {
  params: Promise<{ slug: string }>;
};

export async function GET(_request: Request, { params }: Props) {
  const { slug } = await params;

  // Serve static samples from outputs/ when no DB is configured
  if (slug in STATIC_SAMPLES) {
    const filePath = join(process.cwd(), "outputs", STATIC_SAMPLES[slug]);
    try {
      const data = await readFile(filePath);
      return new Response(data, {
        headers: {
          "content-type": "image/png",
          "cache-control": "public, max-age=3600",
        },
      });
    } catch {
      return new Response("Not found", { status: 404 });
    }
  }

  const artwork = await getArtworkBySlug(slug);

  if (!artwork) {
    return new Response("Not found", { status: 404 });
  }

  const modalResponse = await fetchModalFile(artwork.modalPath);

  return new Response(modalResponse.body, {
    headers: {
      "content-type": artwork.contentType,
      "cache-control": "public, max-age=31536000, immutable",
    },
  });
}

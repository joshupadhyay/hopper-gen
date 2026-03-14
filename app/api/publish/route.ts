import { NextResponse } from "next/server";
import { z } from "zod";

import { publishArtwork } from "@/lib/db";

const requestSchema = z.object({
  generationId: z.string().uuid(),
  title: z.string().trim().min(1).max(120),
  attribution: z.string().trim().min(1).max(120),
});

export async function POST(request: Request) {
  try {
    const payload = requestSchema.parse(await request.json());
    const artwork = await publishArtwork(payload);
    return NextResponse.json({ slug: artwork.slug });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Unable to publish artwork.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

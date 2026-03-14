import Link from "next/link";
import { notFound } from "next/navigation";

import { PublishForm } from "@/components/publish-form";
import { formatDisplayDate } from "@/lib/date";
import { getGeneration } from "@/lib/db";

export const dynamic = "force-dynamic";

type Props = {
  params: Promise<{ generationId: string }>;
};

export default async function GenerationPage({ params }: Props) {
  const { generationId } = await params;
  const generation = await getGeneration(generationId);

  if (!generation) {
    notFound();
  }

  return (
    <main className="detail-shell">
      <section className="art-stage">
        <div className="frame">
          <img
            alt={generation.userPrompt}
            className="art-image"
            src={`/artwork/generated/${generation.id}/image`}
          />
        </div>
        <div className="plaque">
          <span>Generated {formatDisplayDate(generation.createdAt)}</span>
          <span>{generation.width} x {generation.height}</span>
        </div>
      </section>

      <section className="detail-sidebar">
        <p className="eyebrow">Result</p>
        <h1>Review the painting, then decide whether to publish it.</h1>
        <p className="lede">
          The final prompt sent to the model is kept server-side. Published
          works store metadata only and continue to read the image from Modal.
        </p>

        <div className="prompt-summary">
          <strong>Prompt</strong>
          <p>{generation.userPrompt}</p>
        </div>

        <div className="detail-actions">
          <Link className="secondary-link" href="/">
            Generate again
          </Link>
        </div>

        <PublishForm generationId={generation.id} />
      </section>
    </main>
  );
}

"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { FormEvent, useMemo, useState, useTransition } from "react";

import type { PublishedArtworkSummary } from "@/lib/types";

type Props = {
  artworks: PublishedArtworkSummary[];
};

export function HomepageClient({ artworks }: Props) {
  const router = useRouter();
  const [prompt, setPrompt] = useState("");
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();

  const generateLabel = useMemo(() => {
    if (!isPending) {
      return "Generate";
    }
    return statusMessage ?? "Generating";
  }, [isPending, statusMessage]);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setErrorMessage(null);
    setStatusMessage("Warming up model");

    const warmTimer = window.setTimeout(() => {
      setStatusMessage("Generating image");
    }, 900);

    const renderTimer = window.setTimeout(() => {
      setStatusMessage("Rendering result");
    }, 3200);

    startTransition(async () => {
      try {
        const runName = crypto.randomUUID();
        const response = await fetch("/api/generate", {
          method: "POST",
          headers: {
            "content-type": "application/json",
          },
          body: JSON.stringify({ prompt, runName }),
        });

        const data = (await response.json()) as
          | { generationId: string }
          | { error: string };

        if (!response.ok || !("generationId" in data)) {
          setErrorMessage("error" in data ? data.error : "Unable to generate artwork.");
          return;
        }

        router.push(`/generate/${data.generationId}`);
      } catch {
        setErrorMessage("Unable to reach the studio service right now.");
      } finally {
        window.clearTimeout(warmTimer);
        window.clearTimeout(renderTimer);
        setStatusMessage(null);
      }
    });
  }

  return (
    <main className="home-shell">
      <section className="hero-panel">
        <p className="eyebrow">Hopper Studio</p>
        <h1>Quiet paintings from a single line of text.</h1>
        <p className="lede">
          Describe the scene. The studio handles the Hopper framing, light, and
          fixed composition settings behind the curtain.
        </p>

        <form className="prompt-form" onSubmit={onSubmit}>
          <label className="sr-only" htmlFor="prompt">
            Describe your artwork
          </label>
          <textarea
            id="prompt"
            name="prompt"
            placeholder="A diner at dawn with rain on the windows"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            rows={5}
            maxLength={400}
          />
          <div className="prompt-actions">
            <button className="primary-button" type="submit" disabled={isPending}>
              {generateLabel}
            </button>
            <span className="hint">Prompt only. No visible settings in v1.</span>
          </div>
        </form>

        {errorMessage ? <p className="form-error">{errorMessage}</p> : null}
      </section>

      <section className="gallery-panel" aria-label="Published artworks">
        <div className="gallery-header">
          <p className="eyebrow">Published Works</p>
          <p className="gallery-copy">
            Recent pieces from the public gallery. Each opens into a framed
            detail view with a shareable permalink.
          </p>
        </div>

        <div className="gallery-field">
          <div className="gallery-track">
            {[...artworks, ...artworks].map((artwork, index) => (
              <Link
                key={`${artwork.slug}-${index}`}
                className="gallery-card"
                href={`/artwork/${artwork.slug}`}
              >
                <div
                  className="gallery-thumb"
                  style={{
                    backgroundImage: `url(/artwork/${artwork.slug}/image)`,
                  }}
                />
                <div className="gallery-meta">
                  <strong>{artwork.title}</strong>
                  <span>{artwork.attribution}</span>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}

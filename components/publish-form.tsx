"use client";

import { FormEvent, useState, useTransition } from "react";
import { useRouter } from "next/navigation";

type Props = {
  generationId: string;
};

export function PublishForm({ generationId }: Props) {
  const router = useRouter();
  const [title, setTitle] = useState("");
  const [attribution, setAttribution] = useState("");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setErrorMessage(null);

    startTransition(async () => {
      try {
        const response = await fetch("/api/publish", {
          method: "POST",
          headers: {
            "content-type": "application/json",
          },
          body: JSON.stringify({
            generationId,
            title,
            attribution,
          }),
        });

        const data = (await response.json()) as { slug?: string; error?: string };
        if (!response.ok || !data.slug) {
          setErrorMessage(data.error ?? "Publishing failed.");
          return;
        }

        router.push(`/artwork/${data.slug}`);
      } catch {
        setErrorMessage("Publishing failed.");
      }
    });
  }

  return (
    <form className="publish-form" onSubmit={onSubmit}>
      <div className="field">
        <label htmlFor="title">Title</label>
        <input
          id="title"
          name="title"
          value={title}
          onChange={(event) => setTitle(event.target.value)}
          placeholder="Sunday Matinee"
          maxLength={120}
          required
        />
      </div>

      <div className="field">
        <label htmlFor="attribution">Attribution</label>
        <input
          id="attribution"
          name="attribution"
          value={attribution}
          onChange={(event) => setAttribution(event.target.value)}
          placeholder="anonymous"
          maxLength={120}
          required
        />
      </div>

      <button className="primary-button" type="submit" disabled={isPending}>
        {isPending ? "Publishing" : "Publish your artwork"}
      </button>

      {errorMessage ? <p className="form-error">{errorMessage}</p> : null}
    </form>
  );
}

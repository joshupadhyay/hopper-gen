export type GenerationRecord = {
  id: string;
  clientRunId: string;
  userPrompt: string;
  fullPrompt: string;
  modalPath: string;
  contentType: string;
  width: number;
  height: number;
  status: string;
  createdAt: Date;
};

export type ArtworkRecord = {
  id: string;
  generationId: string;
  slug: string;
  title: string;
  attribution: string;
  createdAt: Date;
  publishedAt: Date;
  modalPath: string;
  contentType: string;
  userPrompt: string;
};

export type PublishedArtworkSummary = {
  slug: string;
  title: string;
  attribution: string;
  publishedAt: Date;
};

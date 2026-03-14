import { fetchModalFile } from "@/lib/modal";
import { getGeneration } from "@/lib/db";

type Props = {
  params: Promise<{ generationId: string }>;
};

export async function GET(_request: Request, { params }: Props) {
  const { generationId } = await params;
  const generation = await getGeneration(generationId);

  if (!generation) {
    return new Response("Not found", { status: 404 });
  }

  const modalResponse = await fetchModalFile(generation.modalPath);

  return new Response(modalResponse.body, {
    headers: {
      "content-type": generation.contentType,
      "cache-control": "private, max-age=300",
    },
  });
}

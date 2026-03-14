import { NextResponse } from "next/server";
import { z } from "zod";

import { appConfig } from "@/lib/config";
import { createGeneration } from "@/lib/db";
import { generateViaModal } from "@/lib/modal";
import { isBlankPrompt, prefixPrompt } from "@/lib/prefix";

const requestSchema = z.object({
  prompt: z.string(),
  runName: z.string().uuid(),
});

export async function POST(request: Request) {
  try {
    const payload = requestSchema.parse(await request.json());

    if (isBlankPrompt(payload.prompt)) {
      return NextResponse.json(
        { error: "Please enter a prompt before generating." },
        { status: 400 },
      );
    }

    const fullPrompt = prefixPrompt(payload.prompt);
    const modalResult = await generateViaModal({
      prompt: fullPrompt,
      runName: payload.runName,
      adapterName: appConfig.generationDefaults.adapterName,
      guidanceScale: appConfig.generationDefaults.guidanceScale,
      numSteps: appConfig.generationDefaults.numSteps,
      width: appConfig.generationDefaults.width,
      height: appConfig.generationDefaults.height,
    });

    const generation = await createGeneration({
      clientRunId: payload.runName,
      userPrompt: payload.prompt.trim(),
      fullPrompt,
      modalPath: modalResult.modalPath,
      contentType: modalResult.contentType,
      width: modalResult.width,
      height: modalResult.height,
    });

    return NextResponse.json({ generationId: generation.id });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Generation failed unexpectedly.";

    return NextResponse.json({ error: message }, { status: 500 });
  }
}

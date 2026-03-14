import { appConfig, requireConfigValue } from "@/lib/config";

type ModalGeneratePayload = {
  prompt: string;
  runName: string;
  adapterName: string;
  guidanceScale: number;
  numSteps: number;
  width: number;
  height: number;
};

export type ModalGenerateResult = {
  modalPath: string;
  contentType: string;
  width: number;
  height: number;
};

type FlexibleImageResult = {
  modal_path?: string;
  modalPath?: string;
  content_type?: string;
  contentType?: string;
  width?: number;
  height?: number;
};

type FlexibleResponse = FlexibleImageResult & {
  image?: FlexibleImageResult;
  images?: FlexibleImageResult[];
};

function extractImageResult(data: FlexibleResponse): ModalGenerateResult {
  const candidate = data.images?.[0] ?? data.image ?? data;
  const modalPath = candidate.modalPath ?? candidate.modal_path;

  if (!modalPath) {
    throw new Error("Modal response did not include a modal path");
  }

  return {
    modalPath,
    contentType: candidate.contentType ?? candidate.content_type ?? "image/png",
    width: candidate.width ?? appConfig.generationDefaults.width,
    height: candidate.height ?? appConfig.generationDefaults.height,
  };
}

export async function generateViaModal(
  payload: ModalGeneratePayload,
): Promise<ModalGenerateResult> {
  const modalGenerateUrl = requireConfigValue(
    appConfig.modalGenerateUrl,
    "MODAL_GENERATE_URL",
  );

  const response = await fetch(modalGenerateUrl, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...(appConfig.modalSharedSecret
        ? { "x-modal-shared-secret": appConfig.modalSharedSecret }
        : {}),
    },
    body: JSON.stringify(payload),
    cache: "no-store",
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Modal generation failed (${response.status}): ${body}`);
  }

  const data = (await response.json()) as FlexibleResponse;
  return extractImageResult(data);
}

export async function fetchModalFile(modalPath: string) {
  const modalFileBaseUrl = requireConfigValue(
    appConfig.modalFileBaseUrl,
    "MODAL_FILE_BASE_URL",
  );
  const safePath = modalPath.split("/").map(encodeURIComponent).join("/");
  const response = await fetch(`${modalFileBaseUrl}/${safePath}`, {
    headers: appConfig.modalSharedSecret
      ? { "x-modal-shared-secret": appConfig.modalSharedSecret }
      : undefined,
    cache: "force-cache",
  });

  if (!response.ok) {
    throw new Error(`Unable to load Modal image at ${modalPath}`);
  }

  return response;
}

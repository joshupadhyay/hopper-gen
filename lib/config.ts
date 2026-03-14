function getNumberEnv(name: string, fallback: number) {
  const value = process.env[name];
  if (!value) {
    return fallback;
  }
  const parsed = Number(value);
  if (Number.isNaN(parsed)) {
    throw new Error(`Invalid numeric environment variable: ${name}`);
  }
  return parsed;
}

export const appConfig = {
  databaseUrl: process.env.DATABASE_URL,
  modalGenerateUrl: process.env.MODAL_GENERATE_URL,
  modalFileBaseUrl: process.env.MODAL_FILE_BASE_URL,
  modalSharedSecret: process.env.MODAL_SHARED_SECRET,
  siteUrl: process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000",
  generationDefaults: {
    adapterName: process.env.HOPPER_ADAPTER_NAME ?? "v1",
    guidanceScale: getNumberEnv("HOPPER_GUIDANCE_SCALE", 7.5),
    numSteps: getNumberEnv("HOPPER_NUM_STEPS", 50),
    width: getNumberEnv("HOPPER_WIDTH", 1024),
    height: getNumberEnv("HOPPER_HEIGHT", 1024),
  },
};

export function requireConfigValue(
  value: string | undefined,
  name: keyof typeof appConfig | string,
) {
  if (!value) {
    throw new Error(`Missing environment variable: ${name}`);
  }

  return value;
}

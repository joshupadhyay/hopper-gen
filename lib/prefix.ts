const HOPPER_PREFIX = "hopper style";

export function prefixPrompt(prompt: string) {
  const trimmed = prompt.trim().replace(/\s+/g, " ");
  const normalized = trimmed.replace(/^hopper style[:,\s-]*/i, "").trim();
  return normalized ? `${HOPPER_PREFIX} ${normalized}` : HOPPER_PREFIX;
}

export function isBlankPrompt(prompt: string) {
  return prompt.trim().length === 0;
}

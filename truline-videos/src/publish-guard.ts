/** Minutes of clearance required before first pitch. Mirrors the backend's
 *  PREVIEW_MIN_LEAD_MINUTES — the backend gates selection, this gates upload,
 *  and rendering between them can take minutes. */
export const MIN_LEAD_MINUTES = 45;

export function mayPublish(
  adapterPublishable: boolean,
  minutesToFirstPitch: number,
  minLeadMinutes: number = MIN_LEAD_MINUTES,
): boolean {
  if (!adapterPublishable) return false;
  return minutesToFirstPitch > minLeadMinutes;
}

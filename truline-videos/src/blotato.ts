import axios from 'axios';
import { readFileSync } from 'fs';

const BLOTATO_API = 'https://backend.blotato.com/v2';

export interface BlotatoConfig {
  apiKey: string;
  tiktokAccountId?: string;
  instagramAccountId?: string;
}

export function blotatoConfigFromEnv(env: NodeJS.ProcessEnv = process.env): BlotatoConfig {
  return {
    apiKey: env.BLOTATO_API_KEY || '',
    tiktokAccountId: env.BLOTATO_TIKTOK_ACCOUNT_ID,
    instagramAccountId: env.BLOTATO_INSTAGRAM_ACCOUNT_ID,
  };
}

/**
 * What actually happened to an upload attempt.
 *
 * `uploaded` is true once the file made it into Blotato's media storage.
 * `posted` lists only the platforms that came back with a real
 * postSubmissionId — a resolved promise from this function is NOT by itself
 * proof that anything went out. Blotato being down, no account ID
 * configured, or every platform rejecting the post are all failures that
 * still resolve cleanly; callers MUST check `posted.length > 0` before
 * treating this as a success (e.g. before marking a pick as dedupe-posted).
 */
export interface BlotatoUploadResult {
  uploaded: boolean;
  posted: string[];
}

const NO_RESULT: BlotatoUploadResult = { uploaded: false, posted: [] };

/** Uploads and schedules. A missing apiKey is a DRY RUN, never an error —
 *  that is how both scripts are exercised without posting. Never throws:
 *  every failure mode is reported via the returned `posted` list instead,
 *  so a caller cannot mistake a resolved promise for a successful post. */
export async function uploadToBlotato(
  videoPath: string,
  caption: string,
  cfg: BlotatoConfig,
): Promise<BlotatoUploadResult> {
  if (!cfg.apiKey) {
    console.log('[DRY-RUN] Would upload:', videoPath);
    console.log('[DRY-RUN] Caption:', caption);
    return NO_RESULT;
  }

  const headers = { 'blotato-api-key': cfg.apiKey, 'Content-Type': 'application/json' };
  const filename = videoPath.split('/').pop() || 'video.mp4';

  try {
    const videoData = readFileSync(videoPath);

    const uploadResp = await axios.post(`${BLOTATO_API}/media/uploads`, { filename }, { headers, timeout: 60000 });
    const { presignedUrl, publicUrl } = uploadResp.data;

    await axios.put(presignedUrl, videoData, {
      headers: { 'Content-Type': 'video/mp4' },
      timeout: 60000,
      maxBodyLength: Infinity,
    });

    console.log(`Uploaded: ${publicUrl}`);

    const posted: string[] = [];
    for (const [platform, accountId] of [
      ['tiktok', cfg.tiktokAccountId],
      ['instagram', cfg.instagramAccountId],
    ] as const) {
      if (!accountId) continue;

      const payload = {
        post: {
          accountId,
          content: { text: caption, mediaUrls: [publicUrl], platform },
          target: { targetType: platform },
        },
        useNextFreeSlot: true,
      };

      try {
        const resp = await axios.post(`${BLOTATO_API}/posts`, payload, { headers, timeout: 30000 });
        if (resp.data?.postSubmissionId) {
          console.log(`Posted to ${platform}:`, resp.data.postSubmissionId);
          posted.push(platform);
        } else {
          console.error(`Post to ${platform} returned no postSubmissionId — not counting it as posted:`, resp.data);
        }
      } catch (err: any) {
        console.error(`Failed to post to ${platform}:`, err?.response?.data || err.message);
      }
    }
    return { uploaded: true, posted };
  } catch (err: any) {
    console.error('Upload failed:', err?.response?.data || err.message);
    return NO_RESULT;
  }
}

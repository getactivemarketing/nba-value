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

/** Uploads and schedules. A missing apiKey is a DRY RUN, never an error —
 *  that is how both scripts are exercised without posting. */
export async function uploadToBlotato(
  videoPath: string,
  caption: string,
  cfg: BlotatoConfig,
): Promise<void> {
  if (!cfg.apiKey) {
    console.log('[DRY-RUN] Would upload:', videoPath);
    console.log('[DRY-RUN] Caption:', caption);
    return;
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
        console.log(`Posted to ${platform}:`, resp.data.postSubmissionId);
      } catch (err: any) {
        console.error(`Failed to post to ${platform}:`, err?.response?.data || err.message);
      }
    }
  } catch (err: any) {
    console.error('Upload failed:', err?.response?.data || err.message);
  }
}

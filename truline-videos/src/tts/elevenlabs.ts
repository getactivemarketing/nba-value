import axios from 'axios';
import { writeFileSync, statSync, unlinkSync } from 'fs';
import type { TtsAdapter } from './types';

const VOICE_ID = process.env.ELEVENLABS_VOICE_ID || 'JBFqnCBsd6RMkjVDRZzb';

export function elevenLabsAdapter(apiKey: string): TtsAdapter {
  return Object.freeze<TtsAdapter>({
    id: 'elevenlabs',
    publishable: true,
    async synthesize(text, outPath) {
      const resp = await axios.post(
        `https://api.elevenlabs.io/v1/text-to-speech/${VOICE_ID}`,
        { text, model_id: 'eleven_turbo_v2_5' },
        {
          headers: { 'xi-api-key': apiKey, 'Content-Type': 'application/json' },
          responseType: 'arraybuffer',
          timeout: 60000,
        },
      );
      writeFileSync(outPath, Buffer.from(resp.data));
      const stats = statSync(outPath);
      if (stats.size === 0) {
        unlinkSync(outPath);
        throw new Error('ElevenLabs returned empty audio');
      }
    },
  });
}

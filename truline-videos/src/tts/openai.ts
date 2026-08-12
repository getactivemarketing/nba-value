import axios from 'axios';
import { writeFileSync, statSync, unlinkSync } from 'fs';
import type { TtsAdapter } from './types';

export function openAiAdapter(apiKey: string): TtsAdapter {
  return Object.freeze<TtsAdapter>({
    id: 'openai',
    publishable: true,
    async synthesize(text, outPath) {
      const resp = await axios.post(
        'https://api.openai.com/v1/audio/speech',
        { model: 'gpt-4o-mini-tts', voice: 'onyx', input: text, response_format: 'mp3' },
        {
          headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
          responseType: 'arraybuffer',
          timeout: 60000,
        },
      );
      writeFileSync(outPath, Buffer.from(resp.data));
      const stats = statSync(outPath);
      if (stats.size === 0) {
        unlinkSync(outPath);
        throw new Error('OpenAI returned empty audio');
      }
    },
  });
}

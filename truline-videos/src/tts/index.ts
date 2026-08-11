import { elevenLabsAdapter } from './elevenlabs';
import { openAiAdapter } from './openai';
import { sayAdapter } from './say';
import type { TtsAdapter } from './types';

export type { TtsAdapter };

export function selectAdapter(env: NodeJS.ProcessEnv = process.env): TtsAdapter {
  const explicit = env.TTS_PROVIDER;

  if (explicit === 'elevenlabs') {
    if (!env.ELEVENLABS_API_KEY) throw new Error('TTS_PROVIDER=elevenlabs but ELEVENLABS_API_KEY is unset');
    return elevenLabsAdapter(env.ELEVENLABS_API_KEY);
  }
  if (explicit === 'openai') {
    if (!env.OPENAI_API_KEY) throw new Error('TTS_PROVIDER=openai but OPENAI_API_KEY is unset');
    return openAiAdapter(env.OPENAI_API_KEY);
  }
  if (explicit === 'say') return sayAdapter();

  if (env.ELEVENLABS_API_KEY) return elevenLabsAdapter(env.ELEVENLABS_API_KEY);
  if (env.OPENAI_API_KEY) return openAiAdapter(env.OPENAI_API_KEY);
  return sayAdapter();
}

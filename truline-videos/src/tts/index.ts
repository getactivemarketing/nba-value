import { elevenLabsAdapter } from './elevenlabs';
import { openAiAdapter } from './openai';
import { sayAdapter } from './say';
import type { TtsAdapter } from './types';

export type { TtsAdapter };

export function selectAdapter(env: NodeJS.ProcessEnv = process.env): TtsAdapter {
  const explicit = env.TTS_PROVIDER;

  if (explicit === 'elevenlabs') {
    const key = env.ELEVENLABS_API_KEY?.trim();
    if (!key) throw new Error('TTS_PROVIDER=elevenlabs but ELEVENLABS_API_KEY is unset');
    return elevenLabsAdapter(key);
  }
  if (explicit === 'openai') {
    const key = env.OPENAI_API_KEY?.trim();
    if (!key) throw new Error('TTS_PROVIDER=openai but OPENAI_API_KEY is unset');
    return openAiAdapter(key);
  }
  if (explicit === 'say') return sayAdapter();
  if (explicit) throw new Error(`TTS_PROVIDER="${explicit}" is unrecognised. Valid options: elevenlabs, openai, say`);

  const elevenKey = env.ELEVENLABS_API_KEY?.trim();
  if (elevenKey) return elevenLabsAdapter(elevenKey);
  const openaiKey = env.OPENAI_API_KEY?.trim();
  if (openaiKey) return openAiAdapter(openaiKey);
  return sayAdapter();
}

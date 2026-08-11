import { describe, expect, it } from 'vitest';
import { selectAdapter } from './index';

describe('selectAdapter', () => {
  it('prefers elevenlabs when its key is present', () => {
    expect(selectAdapter({ ELEVENLABS_API_KEY: 'k' }).id).toBe('elevenlabs');
  });

  it('falls back to openai when only that key is present', () => {
    expect(selectAdapter({ OPENAI_API_KEY: 'k' }).id).toBe('openai');
  });

  it('honours an explicit TTS_PROVIDER override', () => {
    const env = { TTS_PROVIDER: 'openai', ELEVENLABS_API_KEY: 'k', OPENAI_API_KEY: 'k' };
    expect(selectAdapter(env).id).toBe('openai');
  });

  it('falls back to say when no key is configured', () => {
    expect(selectAdapter({}).id).toBe('say');
  });

  it('marks say as not publishable', () => {
    expect(selectAdapter({}).publishable).toBe(false);
  });

  it('marks real providers as publishable', () => {
    expect(selectAdapter({ ELEVENLABS_API_KEY: 'k' }).publishable).toBe(true);
    expect(selectAdapter({ OPENAI_API_KEY: 'k' }).publishable).toBe(true);
  });

  it('throws when an explicit provider has no key', () => {
    expect(() => selectAdapter({ TTS_PROVIDER: 'elevenlabs' })).toThrow(/ELEVENLABS_API_KEY/);
  });
});

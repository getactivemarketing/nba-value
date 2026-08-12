import { describe, expect, it, afterEach, vi, beforeEach } from 'vitest';
import { selectAdapter } from './index';
import { writeFileSync, unlinkSync, statSync, existsSync } from 'fs';
import axios from 'axios';
import { elevenLabsAdapter } from './elevenlabs';
import { openAiAdapter } from './openai';

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

  it('treats whitespace-only key as missing and falls back', () => {
    expect(selectAdapter({ ELEVENLABS_API_KEY: '   ' }).id).toBe('say');
    expect(selectAdapter({ ELEVENLABS_API_KEY: '\t' }).publishable).toBe(false);
  });

  it('throws on unrecognised TTS_PROVIDER value', () => {
    expect(() => selectAdapter({ TTS_PROVIDER: 'azure' })).toThrow(/TTS_PROVIDER="azure" is unrecognised/);
    expect(() => selectAdapter({ TTS_PROVIDER: 'Say' })).toThrow(/TTS_PROVIDER="Say" is unrecognised/);
  });

  it('frozen adapters cannot have publishable reassigned', () => {
    const adapter = selectAdapter({ ELEVENLABS_API_KEY: 'k' });
    const descriptor = Object.getOwnPropertyDescriptor(adapter, 'publishable');
    expect(descriptor?.configurable).toBe(false);
    expect(descriptor?.writable).toBe(false);
  });

  it('frozen say adapter keeps publishable=false immutable', () => {
    const adapter = selectAdapter({});
    const descriptor = Object.getOwnPropertyDescriptor(adapter, 'publishable');
    expect(descriptor?.configurable).toBe(false);
    expect(descriptor?.writable).toBe(false);
    expect(adapter.publishable).toBe(false);
  });
});

describe('synthesize - file size validation', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    // Cleanup test files
    const tmpFiles = [
      '/tmp/test-elevenlabs-empty.wav',
      '/tmp/test-elevenlabs-valid.wav',
      '/tmp/test-openai-empty.wav',
      '/tmp/test-openai-valid.wav',
    ];
    tmpFiles.forEach((f) => {
      try {
        unlinkSync(f);
      } catch {
        // ignore
      }
    });
  });

  it('elevenlabs adapter rejects empty responses and deletes the file', async () => {
    const postSpy = vi.spyOn(axios, 'post').mockResolvedValueOnce({
      data: new ArrayBuffer(0), // Empty response
    });

    const adapter = elevenLabsAdapter('test-key');
    const tmpPath = '/tmp/test-elevenlabs-empty.wav';

    await expect(adapter.synthesize('test text', tmpPath)).rejects.toThrow(/empty audio/);

    // Verify file was deleted
    expect(existsSync(tmpPath)).toBe(false);

    postSpy.mockRestore();
  });

  it('elevenlabs adapter accepts non-empty responses and creates file', async () => {
    const audioData = Buffer.from('fake audio data');
    const postSpy = vi.spyOn(axios, 'post').mockResolvedValueOnce({
      data: audioData,
    });

    const adapter = elevenLabsAdapter('test-key');
    const tmpPath = '/tmp/test-elevenlabs-valid.wav';

    await adapter.synthesize('test text', tmpPath);

    // Verify file exists with correct content
    expect(existsSync(tmpPath)).toBe(true);
    const stats = statSync(tmpPath);
    expect(stats.size).toBe(audioData.length);
    expect(stats.size).toBeGreaterThan(0);

    postSpy.mockRestore();
  });

  it('openai adapter rejects empty responses and deletes the file', async () => {
    const postSpy = vi.spyOn(axios, 'post').mockResolvedValueOnce({
      data: new ArrayBuffer(0), // Empty response
    });

    const adapter = openAiAdapter('test-key');
    const tmpPath = '/tmp/test-openai-empty.wav';

    await expect(adapter.synthesize('test text', tmpPath)).rejects.toThrow(/empty audio/);

    // Verify file was deleted
    expect(existsSync(tmpPath)).toBe(false);

    postSpy.mockRestore();
  });

  it('openai adapter accepts non-empty responses and creates file', async () => {
    const audioData = Buffer.from('fake audio data');
    const postSpy = vi.spyOn(axios, 'post').mockResolvedValueOnce({
      data: audioData,
    });

    const adapter = openAiAdapter('test-key');
    const tmpPath = '/tmp/test-openai-valid.wav';

    await adapter.synthesize('test text', tmpPath);

    // Verify file exists with correct content
    expect(existsSync(tmpPath)).toBe(true);
    const stats = statSync(tmpPath);
    expect(stats.size).toBe(audioData.length);
    expect(stats.size).toBeGreaterThan(0);

    postSpy.mockRestore();
  });
});

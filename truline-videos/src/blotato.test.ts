import { describe, expect, it, vi, afterEach } from 'vitest';
import axios from 'axios';
import { readFileSync } from 'fs';
import { blotatoConfigFromEnv, uploadToBlotato } from './blotato';

// Node's built-in `fs` module exports are non-configurable, so vi.spyOn(fs,
// 'readFileSync') throws "Cannot redefine property" — vi.mock is the only
// way to stub it out here.
vi.mock('fs', () => ({ readFileSync: vi.fn() }));
const readFileSyncMock = vi.mocked(readFileSync);

afterEach(() => {
  vi.restoreAllMocks();
  readFileSyncMock.mockReset();
});

const cfg = (over: Partial<ReturnType<typeof blotatoConfigFromEnv>> = {}) => ({
  apiKey: 'key',
  tiktokAccountId: 'tt-1',
  instagramAccountId: 'ig-1',
  ...over,
});

describe('blotatoConfigFromEnv', () => {
  it('reads apiKey and account ids from env', () => {
    const config = blotatoConfigFromEnv({
      BLOTATO_API_KEY: 'k',
      BLOTATO_TIKTOK_ACCOUNT_ID: 'tt',
      BLOTATO_INSTAGRAM_ACCOUNT_ID: 'ig',
    } as NodeJS.ProcessEnv);
    expect(config).toEqual({ apiKey: 'k', tiktokAccountId: 'tt', instagramAccountId: 'ig' });
  });

  it('defaults apiKey to empty string when unset', () => {
    expect(blotatoConfigFromEnv({} as NodeJS.ProcessEnv).apiKey).toBe('');
  });
});

describe('uploadToBlotato — the promise resolving is never proof of a real post', () => {
  it('a missing apiKey is a dry run: resolves with uploaded=false, posted=[]', async () => {
    const postSpy = vi.spyOn(axios, 'post');
    const result = await uploadToBlotato('/tmp/v.mp4', 'caption', cfg({ apiKey: '' }));
    expect(result).toEqual({ uploaded: false, posted: [] });
    expect(postSpy).not.toHaveBeenCalled();
  });

  it('a media-upload failure resolves with uploaded=false, posted=[] — never throws', async () => {
    readFileSyncMock.mockReturnValue(Buffer.from('video'));
    vi.spyOn(axios, 'post').mockRejectedValueOnce(new Error('Blotato is down'));
    const result = await uploadToBlotato('/tmp/v.mp4', 'caption', cfg());
    expect(result).toEqual({ uploaded: false, posted: [] });
  });

  it('a presigned PUT failure resolves with uploaded=false, posted=[] — never throws', async () => {
    readFileSyncMock.mockReturnValue(Buffer.from('video'));
    vi.spyOn(axios, 'post').mockResolvedValueOnce({
      data: { presignedUrl: 'https://s3/upload', publicUrl: 'https://cdn/v.mp4' },
    });
    vi.spyOn(axios, 'put').mockRejectedValueOnce(new Error('S3 rejected the PUT'));
    const result = await uploadToBlotato('/tmp/v.mp4', 'caption', cfg());
    expect(result).toEqual({ uploaded: false, posted: [] });
  });

  it('no account ID configured at all: media uploads but posted stays empty — treated as failure', async () => {
    readFileSyncMock.mockReturnValue(Buffer.from('video'));
    const postSpy = vi.spyOn(axios, 'post').mockResolvedValueOnce({
      data: { presignedUrl: 'https://s3/upload', publicUrl: 'https://cdn/v.mp4' },
    });
    vi.spyOn(axios, 'put').mockResolvedValueOnce({});
    const result = await uploadToBlotato(
      '/tmp/v.mp4', 'caption', cfg({ tiktokAccountId: undefined, instagramAccountId: undefined }),
    );
    expect(result.posted).toEqual([]);
    // Only the media upload POST happened — no /posts call was attempted for either platform.
    expect(postSpy).toHaveBeenCalledTimes(1);
  });

  it('a platform rejecting the post is not counted, but a sibling platform succeeding still is', async () => {
    readFileSyncMock.mockReturnValue(Buffer.from('video'));
    vi.spyOn(axios, 'post')
      .mockResolvedValueOnce({ data: { presignedUrl: 'https://s3/upload', publicUrl: 'https://cdn/v.mp4' } })
      .mockRejectedValueOnce(new Error('TikTok rate limited'))
      .mockResolvedValueOnce({ data: { postSubmissionId: 'ig-sub-1' } });
    vi.spyOn(axios, 'put').mockResolvedValueOnce({});
    const result = await uploadToBlotato('/tmp/v.mp4', 'caption', cfg());
    expect(result.posted).toEqual(['instagram']);
  });

  it('a 200 response with no postSubmissionId is not counted as posted', async () => {
    readFileSyncMock.mockReturnValue(Buffer.from('video'));
    vi.spyOn(axios, 'post')
      .mockResolvedValueOnce({ data: { presignedUrl: 'https://s3/upload', publicUrl: 'https://cdn/v.mp4' } })
      .mockResolvedValueOnce({ data: {} }) // tiktok: no postSubmissionId
      .mockResolvedValueOnce({ data: {} }); // instagram: no postSubmissionId
    vi.spyOn(axios, 'put').mockResolvedValueOnce({});
    const result = await uploadToBlotato('/tmp/v.mp4', 'caption', cfg());
    expect(result.posted).toEqual([]);
  });

  it('both platforms confirming are both reported as posted', async () => {
    readFileSyncMock.mockReturnValue(Buffer.from('video'));
    vi.spyOn(axios, 'post')
      .mockResolvedValueOnce({ data: { presignedUrl: 'https://s3/upload', publicUrl: 'https://cdn/v.mp4' } })
      .mockResolvedValueOnce({ data: { postSubmissionId: 'tt-1' } })
      .mockResolvedValueOnce({ data: { postSubmissionId: 'ig-1' } });
    vi.spyOn(axios, 'put').mockResolvedValueOnce({});
    const result = await uploadToBlotato('/tmp/v.mp4', 'caption', cfg());
    expect(result.uploaded).toBe(true);
    expect(result.posted.sort()).toEqual(['instagram', 'tiktok']);
  });
});

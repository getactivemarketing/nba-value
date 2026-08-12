import { execFileSync } from 'child_process';
import { statSync, unlinkSync } from 'fs';
import type { TtsAdapter } from './types';

/**
 * macOS `say`. Offline dev fallback so the pipeline is testable without a key.
 * publishable=false — it is too robotic to publish, and the orchestrator
 * refuses to upload renders narrated with it.
 */
export function sayAdapter(): TtsAdapter {
  return Object.freeze<TtsAdapter>({
    id: 'say',
    publishable: false,
    async synthesize(text, outPath) {
      execFileSync('say', ['-v', 'Samantha', '-o', outPath, '--data-format=LEF32@22050', text]);
      const stats = statSync(outPath);
      if (stats.size === 0) {
        unlinkSync(outPath);
        throw new Error('say produced empty audio');
      }
    },
  });
}

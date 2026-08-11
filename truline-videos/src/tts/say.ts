import { execFileSync } from 'child_process';
import type { TtsAdapter } from './types';

/**
 * macOS `say`. Offline dev fallback so the pipeline is testable without a key.
 * publishable=false — it is too robotic to publish, and the orchestrator
 * refuses to upload renders narrated with it.
 */
export function sayAdapter(): TtsAdapter {
  return {
    id: 'say',
    publishable: false,
    async synthesize(text, outPath) {
      execFileSync('say', ['-v', 'Samantha', '-o', outPath, '--data-format=LEF32@22050', text]);
    },
  };
}

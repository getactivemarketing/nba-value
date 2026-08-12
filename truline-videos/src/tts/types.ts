export interface TtsAdapter {
  readonly id: 'elevenlabs' | 'openai' | 'say';
  /** False means renders using this adapter must never be uploaded. */
  readonly publishable: boolean;
  synthesize(text: string, outPath: string): Promise<void>;
}

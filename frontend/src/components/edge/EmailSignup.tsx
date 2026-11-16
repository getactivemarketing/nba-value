// frontend/src/components/edge/EmailSignup.tsx
import { useState, type FormEvent } from 'react';
import { edgeApi, type SignupSource } from '@/lib/edgeApi';

type Status = 'idle' | 'sending' | 'done' | 'error';

export function EmailSignup({ source }: { source: SignupSource }) {
  const [email, setEmail] = useState('');
  const [website, setWebsite] = useState(''); // honeypot: bots fill it, people never see it
  const [status, setStatus] = useState<Status>('idle');

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setStatus('sending');
    try {
      await edgeApi.signup(email.trim(), source, website);
      setStatus('done');
    } catch {
      setStatus('error');
    }
  }

  return (
    <section className="mt-10 bg-[#0b0e14] rounded-lg p-5 border border-[#1e293b]">
      <h2 className="text-base font-black font-mono text-[#f1f5f9]">Get TruLine Edge updates</h2>
      {status === 'done' ? (
        <p className="mt-2 text-sm font-mono text-[#4ade80]">You're on the list. Thanks.</p>
      ) : (
        <>
          <p className="mt-1 text-xs font-mono text-[#94a3b8]">
            New tools, new sports and launch offers by email. No spam, unsubscribe anytime.
          </p>
          <form onSubmit={onSubmit} className="mt-3 flex flex-col sm:flex-row gap-2">
            <label htmlFor={`signup-${source}`} className="sr-only">Email address</label>
            <input id={`signup-${source}`} type="email" required autoComplete="email" value={email}
                   onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com"
                   className="flex-1 bg-[#05070a] border border-[#1e293b] rounded px-3 py-2 text-sm font-mono
                              text-[#f1f5f9] placeholder:text-[#475569] focus:outline-none focus:border-[#a4e6ff]" />
            <div aria-hidden="true" className="absolute -left-[9999px] w-px h-px overflow-hidden">
              <input name="website" tabIndex={-1} autoComplete="off" value={website}
                     onChange={(e) => setWebsite(e.target.value)} />
            </div>
            <button type="submit" disabled={status === 'sending'}
                    className="bg-[#a4e6ff] text-[#05070a] font-mono font-bold text-sm rounded px-4 py-2
                               disabled:opacity-60">
              {status === 'sending' ? 'Sending…' : 'Notify me'}
            </button>
          </form>
          {status === 'error' && (
            <p role="alert" className="mt-2 text-xs font-mono text-[#f87171]">
              That didn't go through. Check the address and try again.
            </p>
          )}
        </>
      )}
    </section>
  );
}

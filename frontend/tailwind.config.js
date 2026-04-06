/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        tru: {
          bg: '#0a0e17',
          surface: '#111827',
          card: '#151d2e',
          border: '#1e293b',
          'border-light': '#2a3a52',
        },
        accent: {
          cyan: '#06d6a0',
          'cyan-dim': '#06d6a033',
          blue: '#3b82f6',
        },
        value: {
          hot: '#10b981',
          'hot-bg': '#10b98115',
          warm: '#f59e0b',
          'warm-bg': '#f59e0b15',
          cold: '#64748b',
        },
        win: '#10b981',
        loss: '#ef4444',
        push: '#f59e0b',
        txt: {
          primary: '#f1f5f9',
          secondary: '#94a3b8',
          muted: '#64748b',
        },
      },
      fontFamily: {
        display: ['"DM Sans"', 'system-ui', 'sans-serif'],
        body: ['"DM Sans"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      animation: {
        'glow': 'glow 2s ease-in-out infinite alternate',
        'slide-up': 'slideUp 0.4s ease-out',
        'fade-in': 'fadeIn 0.3s ease-out',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px rgba(6, 214, 160, 0.1)' },
          '100%': { boxShadow: '0 0 20px rgba(6, 214, 160, 0.15)' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}

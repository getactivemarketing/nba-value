/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Value Score colors
        value: {
          high: '#10B981',    // Green - High value (80+)
          medium: '#F59E0B',  // Amber - Medium value (50-79)
          low: '#6B7280',     // Gray - Low value (<50)
        },
        // Edge colors
        edge: {
          positive: '#10B981',
          negative: '#EF4444',
        },
      },
    },
  },
  plugins: [],
};

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        risk: {
          low: '#0f766e',
          medium: '#b45309',
          high: '#b91c1c',
        },
      },
    },
  },
  plugins: [],
};

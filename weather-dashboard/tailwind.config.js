/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'neo-yellow': '#FFD23F',
        'neo-pink': '#FF6B6B',
        'neo-blue': '#74B9FF',
        'neo-green': '#88D498',
        'neo-bg': '#FFFDF5',
        'neo-black': '#000000',
        'neo-white': '#ffffff',
      },
      boxShadow: {
        'neo': '5px 5px 0 0 #000',
        'neo-sm': '3px 3px 0 0 #000',
        'neo-lg': '8px 8px 0 0 #000',
      },
      borderWidth: {
        '3': '3px',
      },
      fontFamily: {
        'display': ['Syne', 'sans-serif'],
        'heading': ['Space Grotesk', 'sans-serif'],
        'body': ['Inter', 'sans-serif'],
        'mono': ['Space Mono', 'monospace'],
      }
    },
  },
  plugins: [],
}

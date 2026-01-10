/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./pages/**/*.{js,jsx}",
    "./components/**/*.{js,jsx}"
  ],
  theme: {
    extend: {
      colors: {
        accent: "var(--accent)",
        muted: "var(--muted)",
        card: "var(--card)"
      },
      borderRadius: {
        xl: "12px"
      }
    }
  },
  plugins: []
};
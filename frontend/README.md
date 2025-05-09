# CALM React Frontend

This is the frontend for the CALM (Climate Assessment & Logging Monitor) application, built with React.

## Features

- Modern, responsive UI for desktop and mobile
- Interactive tornado risk map using Leaflet
- Real-time weather monitoring and prediction
- Visual statistics using Chart.js
- AI model training interface

## Tech Stack

- React 18
- React Router for navigation
- Axios for API communication
- Bootstrap 5 for styling
- Chart.js for data visualization
- Leaflet for interactive maps

## Development Setup

### Prerequisites

- Node.js (v14+)
- npm or yarn

### Installation

1. Install dependencies:
   ```
   npm install
   ```

2. Start the development server:
   ```
   npm start
   ```

3. The app will be available at http://localhost:3000

## Building for Production

```
npm run build
```

This will create an optimized production build in the `build` folder, which can be served by the Flask backend.

## Project Structure

- `/src` - Source code
  - `/components` - Reusable UI components
  - `/pages` - Page components
  - `/services` - API services
  - `/assets` - Static assets

## Integration with Flask Backend

This React app communicates with the Flask backend API. In development, it proxies requests to http://localhost:5000.

## Browser Compatibility

The app is designed to work on all modern browsers including:
- Chrome
- Firefox
- Safari
- Edge
- Mobile browsers (iOS Safari, Android Chrome)

## PWA Support

The app includes Progressive Web App features allowing it to be installed on mobile devices. 
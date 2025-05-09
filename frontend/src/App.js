import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import weatherService from './services/api';

// Pages
import Home from './pages/Home';
import TornadoMap from './pages/TornadoMap';
import Stats from './pages/Stats';
import About from './pages/About';
import Model from './pages/Model';

// Components
import Loader from './components/Loader';

function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [apiStatus, setApiStatus] = useState('checking');

  useEffect(() => {
    // Check API health on initial load
    const checkApiHealth = async () => {
      try {
        await weatherService.healthCheck();
        setApiStatus('connected');
      } catch (error) {
        setApiStatus('disconnected');
        console.error('API not available:', error);
      } finally {
        setIsLoading(false);
      }
    };

    checkApiHealth();
  }, []);

  if (isLoading) {
    return <Loader />;
  }

  return (
    <Router>
      <div className="app-container">
        <div className="ai-version-banner">
          AI-Powered Tornado Prediction Platform
        </div>
        
        <header className="app-header" style={{ marginTop: '28px' }}>
          <h1 className="app-title">CALM</h1>
          <nav className="nav-links">
            <NavLink to="/" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'} end>
              <i className="bi bi-house-door me-2"></i> Home
            </NavLink>
            <NavLink to="/tornado" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
              <i className="bi bi-tornado me-2"></i> Tornado Risk
            </NavLink>
            <NavLink to="/stats" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
              <i className="bi bi-graph-up me-2"></i> Stats
            </NavLink>
            <NavLink to="/model" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
              <i className="bi bi-cpu me-2"></i> Model
            </NavLink>
            <NavLink to="/about" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
              <i className="bi bi-info-circle me-2"></i> About
            </NavLink>
          </nav>
          <div className="api-status" style={{ fontSize: '0.8rem', color: apiStatus === 'connected' ? 'var(--success)' : 'var(--danger)' }}>
            <i className={`bi ${apiStatus === 'connected' ? 'bi-check-circle-fill' : 'bi-exclamation-triangle-fill'} me-1`}></i>
            API {apiStatus === 'connected' ? 'Connected' : 'Disconnected'}
          </div>
        </header>

        <main className="app-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/tornado" element={<TornadoMap />} />
            <Route path="/stats" element={<Stats />} />
            <Route path="/model" element={<Model />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>

        <footer className="app-footer text-center p-3 mt-4" style={{ borderTop: '1px solid var(--border)', color: 'var(--text-secondary)' }}>
          <div className="container">
            <div className="row">
              <div className="col-md-6 text-md-start">
                <small>CALM: Climate Assessment & Logging Monitor</small>
              </div>
              <div className="col-md-6 text-md-end">
                <small>API Status: <span className={apiStatus === 'connected' ? 'text-success' : 'text-danger'}>
                  {apiStatus === 'connected' ? 'Connected' : 'Disconnected'}
                </span></small>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </Router>
  );
}

export default App; 
import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import weatherService from '../services/api';
import Loader from '../components/Loader';

const Home = () => {
  const [location, setLocation] = useState('');
  const [weatherData, setWeatherData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!location) return;

    setIsLoading(true);
    setError(null);
    
    try {
      const data = await weatherService.getPredictions({ location });
      setWeatherData(data);
    } catch (err) {
      setError('Failed to get weather data. Please try again.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleUseCurrentLocation = () => {
    if (navigator.geolocation) {
      setIsLoading(true);
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          try {
            const { latitude, longitude } = position.coords;
            const data = await weatherService.getPredictions({ 
              lat: latitude,
              lon: longitude
            });
            setWeatherData(data);
          } catch (err) {
            setError('Failed to get weather data. Please try again.');
            console.error(err);
          } finally {
            setIsLoading(false);
          }
        },
        (err) => {
          setError('Unable to get your location. Please allow location access or enter a location manually.');
          setIsLoading(false);
          console.error(err);
        }
      );
    } else {
      setError('Geolocation is not supported by your browser.');
    }
  };

  return (
    <div>
      <section className="hero">
        <div className="container">
          <h1>Advanced Climate Monitoring</h1>
          <p>Using artificial intelligence to predict and track severe weather events with unprecedented accuracy.</p>
          <div className="row justify-content-center">
            <div className="col-md-6">
              <form onSubmit={handleSearch} className="d-flex mb-3">
                <input
                  type="text"
                  className="form-control form-control-lg me-2"
                  placeholder="Enter ZIP code or city name"
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                />
                <button type="submit" className="btn btn-primary btn-lg">
                  <i className="bi bi-search"></i>
                </button>
              </form>
              <button
                onClick={handleUseCurrentLocation}
                className="btn btn-outline-secondary btn-block w-100"
              >
                <i className="bi bi-geo-alt me-2"></i>
                Use My Current Location
              </button>
            </div>
          </div>
        </div>
      </section>

      {isLoading ? (
        <div className="text-center py-5">
          <Loader message="Fetching weather data..." />
        </div>
      ) : error ? (
        <div className="alert alert-danger">{error}</div>
      ) : weatherData ? (
        <div className="weather-results">
          <div className="container">
            <div className="row">
              <div className="col-md-6">
                <div className="card mb-4">
                  <div className="card-body">
                    <h2>{weatherData.location?.name || 'Unknown Location'}</h2>
                    <div className="d-flex align-items-center">
                      <div className="weather-icon me-3">
                        <i className={`fa-solid fa-${weatherData.current?.icon || 'cloud'} fa-3x`}></i>
                      </div>
                      <div>
                        <h1 className="display-4">{weatherData.current?.temp}°F</h1>
                        <p>{weatherData.current?.description}</p>
                      </div>
                    </div>
                    
                    <div className="weather-details mt-4">
                      <div className="row">
                        <div className="col-6">
                          <p><i className="bi bi-droplet me-2"></i> Humidity: {weatherData.current?.humidity}%</p>
                          <p><i className="bi bi-wind me-2"></i> Wind: {weatherData.current?.wind_speed} mph</p>
                        </div>
                        <div className="col-6">
                          <p><i className="bi bi-thermometer-half me-2"></i> Feels like: {weatherData.current?.feels_like}°F</p>
                          <p><i className="bi bi-compass me-2"></i> Pressure: {weatherData.current?.pressure} hPa</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="col-md-6">
                <div className="card mb-4">
                  <div className="card-body">
                    <h3 className="card-title">Severe Weather Risk</h3>
                    {weatherData.risk ? (
                      <div className={`alert alert-${weatherData.risk.level === 'low' ? 'success' : weatherData.risk.level === 'medium' ? 'warning' : 'danger'}`}>
                        <h4 className="alert-heading">
                          <i className={`bi bi-${weatherData.risk.level === 'low' ? 'check-circle' : weatherData.risk.level === 'medium' ? 'exclamation-triangle' : 'x-octagon'} me-2`}></i>
                          {weatherData.risk.level === 'low' ? 'Low' : weatherData.risk.level === 'medium' ? 'Medium' : 'High'} Risk
                        </h4>
                        <p>{weatherData.risk.message}</p>
                        {weatherData.risk.level !== 'low' && (
                          <Link to="/tornado" className="btn btn-primary btn-sm mt-2">
                            View Detailed Risk Map
                          </Link>
                        )}
                      </div>
                    ) : (
                      <p>No severe weather risk data available.</p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="container mt-5">
          <div className="row">
            <div className="col-md-4 mb-4">
              <div className="feature-card">
                <div className="card-body p-4">
                  <div className="feature-icon mb-3">
                    <i className="bi bi-tornado text-primary" style={{ fontSize: '2.5rem' }}></i>
                  </div>
                  <h3 className="mb-3">Tornado Predictions</h3>
                  <p className="text-muted mb-4">
                    Our advanced AI model predicts tornado formation with 85% accuracy up to 72 hours in advance.
                  </p>
                  <Link to="/tornado" className="btn btn-primary">
                    View Tornado Risk
                  </Link>
                </div>
              </div>
            </div>
            
            <div className="col-md-4 mb-4">
              <div className="feature-card">
                <div className="card-body p-4">
                  <div className="feature-icon mb-3">
                    <i className="bi bi-graph-up text-success" style={{ fontSize: '2.5rem' }}></i>
                  </div>
                  <h3 className="mb-3">Performance Stats</h3>
                  <p className="text-muted mb-4">
                    See our model's prediction accuracy and verification against NOAA data.
                  </p>
                  <Link to="/stats" className="btn btn-success">
                    View Stats
                  </Link>
                </div>
              </div>
            </div>
            
            <div className="col-md-4 mb-4">
              <div className="feature-card">
                <div className="card-body p-4">
                  <div className="feature-icon mb-3">
                    <i className="bi bi-cpu text-info" style={{ fontSize: '2.5rem' }}></i>
                  </div>
                  <h3 className="mb-3">AI Model</h3>
                  <p className="text-muted mb-4">
                    Learn about our neural network architecture and how it processes weather data.
                  </p>
                  <Link to="/model" className="btn btn-info">
                    View Model Details
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Home; 
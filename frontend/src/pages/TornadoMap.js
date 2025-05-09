import React, { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Circle, useMap, LayersControl, ImageOverlay } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import weatherService from '../services/api';
import Loader from '../components/Loader';
import L from 'leaflet';

// Fix Leaflet marker icons
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png'
});

// Custom icons for different alert types
const alertIcon = new L.Icon({
  iconUrl: 'https://cdn.jsdelivr.net/gh/pointhi/leaflet-color-markers@master/img/marker-icon-2x-red.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41]
});

const riskColors = {
  low: '#10b981',    // Green
  medium: '#f59e0b', // Yellow/Orange
  high: '#ef4444',   // Red
  extreme: '#7f1d1d' // Dark Red
};

// Component to auto update map on data changes
const MapUpdater = ({ center, zoom }) => {
  const map = useMap();
  
  useEffect(() => {
    if (center && zoom && Array.isArray(center) && center.length === 2 && 
        !isNaN(center[0]) && !isNaN(center[1])) {
      map.setView(center, zoom);
    }
  }, [center, zoom, map]);
  
  return null;
};

// Radar overlay component
const RadarOverlay = ({ isVisible, opacity, radarUrl, bounds }) => {
  if (!isVisible || !radarUrl) return null;
  
  // Validate bounds
  const isValidBounds = bounds && 
                       Array.isArray(bounds) && 
                       bounds.length === 2 && 
                       Array.isArray(bounds[0]) && 
                       Array.isArray(bounds[1]) &&
                       bounds[0].length === 2 && 
                       bounds[1].length === 2 &&
                       !bounds[0].some(isNaN) && 
                       !bounds[1].some(isNaN);
  
  if (!isValidBounds) return null;
  
  return (
    <ImageOverlay 
      url={radarUrl} 
      bounds={bounds} 
      opacity={opacity} 
    />
  );
};

const TornadoMap = () => {
  const [predictions, setPredictions] = useState([]);
  const [nwsAlerts, setNwsAlerts] = useState([]);
  const [isLoadingPredictions, setIsLoadingPredictions] = useState(true);
  const [isLoadingAlerts, setIsLoadingAlerts] = useState(true);
  const [isLoadingRadar, setIsLoadingRadar] = useState(false);
  const [isRunningAnalysis, setIsRunningAnalysis] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(new Date());
  const [error, setError] = useState(null);
  const [mapCenter, setMapCenter] = useState([39.8283, -98.5795]); // Center of US
  const [mapZoom, setMapZoom] = useState(4);
  const [showRadar, setShowRadar] = useState(true);
  const [radarOpacity, setRadarOpacity] = useState(0.7);
  const [radarUrl, setRadarUrl] = useState('');
  const [radarBounds, setRadarBounds] = useState([
    [24.396308, -125.000000], // Southwest coordinates
    [49.384358, -66.934570]   // Northeast coordinates
  ]);
  const [predictionMode, setPredictionMode] = useState('fast'); // 'fast' or 'complete'
  
  const timerRef = useRef(null);
  const radarRefreshRef = useRef(null);

  // Function to fetch tornado predictions
  const fetchPredictions = async (mode = predictionMode) => {
    try {
      setIsLoadingPredictions(true);
      setError(null); // Clear any previous errors
      
      console.log(`Fetching tornado predictions in ${mode} mode...`);
      
      // Add a timeout promise to cancel the request after 60 seconds
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error('Request timed out after 60 seconds')), 60000);
      });
      
      // Race between the actual API call and the timeout
      const data = await Promise.race([
        weatherService.getTornadoPredictions({ 
          useFastMode: mode === 'fast'
        }),
        timeoutPromise
      ]);
      
      console.log("Tornado predictions received:", data);
      
      if (data && data.predictions) {
        // Validate predictions
        const validPredictions = data.predictions.filter(
          pred => pred && isValidLatLng(pred.lat, pred.lon)
        );
        
        setPredictions(validPredictions);
        
        // If there are validated predictions, center the map on the first one
        if (validPredictions.length > 0) {
          const firstPrediction = validPredictions[0];
          setMapCenter([firstPrediction.lat, firstPrediction.lon]);
          setMapZoom(6);
        }
      }
    } catch (err) {
      console.error("Failed to load tornado predictions:", err);
      let errorMessage = 'Failed to load tornado predictions. ';
      
      if (err.message && err.message.includes('timeout')) {
        errorMessage += 'The request timed out. Please try again later.';
      } else if (err.response && err.response.status) {
        errorMessage += `Server returned status ${err.response.status}.`;
      } else {
        errorMessage += 'Please try again later.';
      }
      
      setError(errorMessage);
    } finally {
      setIsLoadingPredictions(false);
      setLastUpdated(new Date());
    }
  };

  // Function to fetch NWS alerts
  const fetchNwsAlerts = async () => {
    try {
      setIsLoadingAlerts(true);
      const data = await weatherService.getWeatherAlerts();
      if (data && data.alerts) {
        // Validate alerts
        const validAlerts = data.alerts.filter(
          alert => alert && isValidLatLng(alert.lat, alert.lon)
        );
        setNwsAlerts(validAlerts);
      }
    } catch (err) {
      console.error('Failed to load NWS alerts:', err);
      // We don't set an error here to avoid blocking the map display
    } finally {
      setIsLoadingAlerts(false);
    }
  };

  // Function to fetch radar data
  const fetchRadarData = async () => {
    try {
      setIsLoadingRadar(true);
      
      // Use NOAA's radar API or your backend API
      // Option 1: Use your backend if it serves radar images
      const data = await weatherService.getLiveRadar();
      if (data && data.radar_url) {
        setRadarUrl(data.radar_url);
        if (data.bounds) {
          setRadarBounds(data.bounds);
        }
      } else {
        // Option 2: Fallback to using direct NOAA/NWS radar tiles
        const timestamp = new Date().getTime();
        setRadarUrl(`https://mesonet.agron.iastate.edu/cgi-bin/wms/nexrad/n0r.cgi?&SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1&LAYERS=nexrad-n0r&STYLES=&FORMAT=image/png&TRANSPARENT=true&HEIGHT=600&WIDTH=800&SRS=EPSG:4326&BBOX=${radarBounds[0][1]},${radarBounds[0][0]},${radarBounds[1][1]},${radarBounds[1][0]}&_=${timestamp}`);
      }
    } catch (err) {
      console.error('Failed to load radar data:', err);
      // Fallback to using direct NOAA/NWS radar tiles
      const timestamp = new Date().getTime();
      setRadarUrl(`https://mesonet.agron.iastate.edu/cgi-bin/wms/nexrad/n0r.cgi?&SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1&LAYERS=nexrad-n0r&STYLES=&FORMAT=image/png&TRANSPARENT=true&HEIGHT=600&WIDTH=800&SRS=EPSG:4326&BBOX=${radarBounds[0][1]},${radarBounds[0][0]},${radarBounds[1][1]},${radarBounds[1][0]}&_=${timestamp}`);
    } finally {
      setIsLoadingRadar(false);
    }
  };

  // Function to run AI predictive analysis
  const runAiAnalysis = async () => {
    try {
      setIsRunningAnalysis(true);
      // Get the latest radar data
      const radarData = await weatherService.getLiveRadar();
      
      // Run the analysis
      const analysisResult = await weatherService.predictWithAI({
        radar_data: radarData,
        fast_mode: predictionMode === 'fast',
        coverage: 'national',  // Explicitly request national coverage
        country: 'US'          // Specify country coverage
      });
      
      // Update the predictions with new data
      if (analysisResult && analysisResult.predictions) {
        setPredictions(prevPredictions => {
          // Merge new predictions with existing ones, avoiding duplicates
          const existingIds = new Set(prevPredictions.map(p => p.id));
          const newPredictions = analysisResult.predictions.filter(p => !existingIds.has(p.id));
          return [...prevPredictions, ...newPredictions];
        });
      }
      
      console.log('AI analysis completed:', new Date().toLocaleString());
    } catch (err) {
      console.error('AI analysis failed:', err);
    } finally {
      setIsRunningAnalysis(false);
      setLastUpdated(new Date());
    }
  };

  // Setup initial data loading and periodic updates
  useEffect(() => {
    // Load initial data
    fetchPredictions();
    fetchNwsAlerts();
    fetchRadarData();
    
    // Set up periodic updates (every 5 minutes)
    timerRef.current = setInterval(() => {
      console.log('Running scheduled update:', new Date().toLocaleString());
      fetchNwsAlerts();
      runAiAnalysis();
    }, 5 * 60 * 1000); // 5 minutes
    
    // Set up radar refresh (every 10 minutes)
    radarRefreshRef.current = setInterval(() => {
      console.log('Refreshing radar data:', new Date().toLocaleString());
      fetchRadarData();
    }, 10 * 60 * 1000); // 10 minutes
    
    // Cleanup on component unmount
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      if (radarRefreshRef.current) {
        clearInterval(radarRefreshRef.current);
      }
    };
  }, []);

  const getRiskColor = (riskLevel) => {
    return riskColors[riskLevel] || riskColors.low;
  };

  const getCircleRadius = (riskLevel) => {
    // Radius in meters
    switch (riskLevel) {
      case 'extreme': return 50000; // 50km
      case 'high': return 40000;    // 40km
      case 'medium': return 30000;  // 30km
      default: return 20000;        // 20km for low
    }
  };

  const getAlertSeverityColor = (severity) => {
    switch (severity.toLowerCase()) {
      case 'extreme': return '#7f1d1d'; // Dark Red
      case 'severe': return '#ef4444';  // Red
      case 'moderate': return '#f59e0b'; // Yellow/Orange
      default: return '#10b981';        // Green
    }
  };

  const isValidLatLng = (lat, lng) => {
    return lat !== undefined && lng !== undefined && 
           !isNaN(lat) && !isNaN(lng) &&
           lat >= -90 && lat <= 90 && lng >= -180 && lng <= 180;
  };

  if (isLoadingPredictions && isLoadingAlerts) {
    return <Loader message="Loading tornado risk map..." />;
  }

  if (error) {
    return (
      <div className="container py-5">
        <div className="alert alert-danger">
          <h4 className="alert-heading">Error Loading Tornado Data</h4>
          <p>{error}</p>
          <hr />
          <p className="mb-0">
            The app may be experiencing connectivity issues with the prediction server.
          </p>
        </div>
        <button 
          className="btn btn-primary" 
          onClick={() => {
            setError(null);
            fetchPredictions();
            fetchNwsAlerts();
          }}
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="tornado-map-container">
      <div className="row mb-4">
        <div className="col-12">
          <h2 className="mb-3">Tornado Risk Map</h2>
          <p className="text-muted">
            This map shows current tornado risk predictions based on our AI model and NWS data.
            Areas are color-coded by risk level, with larger circles indicating higher confidence.
          </p>
          {isRunningAnalysis && (
            <div className="alert alert-info d-flex align-items-center">
              <div className="spinner-border spinner-border-sm me-2" role="status">
                <span className="visually-hidden">Loading...</span>
              </div>
              <span>Running AI predictive analysis... This may take a moment.</span>
            </div>
          )}
        </div>
      </div>

      <div className="row mb-4">
        <div className="col-md-3 mb-3">
          <div className="card">
            <div className="card-body">
              <h5 className="card-title">Risk Legend</h5>
              <div className="d-flex align-items-center mb-2">
                <div style={{ width: 20, height: 20, backgroundColor: riskColors.low, borderRadius: '50%', marginRight: 10 }}></div>
                <span>Low Risk</span>
              </div>
              <div className="d-flex align-items-center mb-2">
                <div style={{ width: 20, height: 20, backgroundColor: riskColors.medium, borderRadius: '50%', marginRight: 10 }}></div>
                <span>Medium Risk</span>
              </div>
              <div className="d-flex align-items-center mb-2">
                <div style={{ width: 20, height: 20, backgroundColor: riskColors.high, borderRadius: '50%', marginRight: 10 }}></div>
                <span>High Risk</span>
              </div>
              <div className="d-flex align-items-center">
                <div style={{ width: 20, height: 20, backgroundColor: riskColors.extreme, borderRadius: '50%', marginRight: 10 }}></div>
                <span>Extreme Risk</span>
              </div>
            </div>
          </div>
          
          <div className="card mt-3">
            <div className="card-body">
              <h5 className="card-title">Prediction Settings</h5>
              <div className="form-check">
                <input
                  className="form-check-input"
                  type="radio"
                  name="predictionMode"
                  id="fastMode"
                  value="fast"
                  checked={predictionMode === 'fast'}
                  onChange={() => setPredictionMode('fast')}
                />
                <label className="form-check-label" htmlFor="fastMode">
                  Fast Mode (Quick Predictions)
                </label>
              </div>
              <div className="form-check">
                <input
                  className="form-check-input"
                  type="radio"
                  name="predictionMode"
                  id="completeMode"
                  value="complete"
                  checked={predictionMode === 'complete'}
                  onChange={() => setPredictionMode('complete')}
                />
                <label className="form-check-label" htmlFor="completeMode">
                  Complete Mode (Full Analysis)
                </label>
              </div>
              <button 
                className="btn btn-sm btn-primary mt-3" 
                onClick={() => fetchPredictions(predictionMode)}
                disabled={isLoadingPredictions}
              >
                {isLoadingPredictions ? (
                  <>
                    <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                    Loading...
                  </>
                ) : (
                  <>Update Predictions</>
                )}
              </button>
            </div>
          </div>

          <div className="card mt-3">
            <div className="card-body">
              <h5 className="card-title">Active Alerts</h5>
              <p className="card-text">
                <strong>{predictions.length}</strong> active tornado risk areas
              </p>
              <p className="card-text">
                <strong>{nwsAlerts.length}</strong> active NWS weather alerts
              </p>
              <p className="card-text small text-muted">
                Last updated: {lastUpdated.toLocaleString()}
              </p>
              <button 
                className="btn btn-sm btn-primary mt-2" 
                onClick={() => {
                  fetchNwsAlerts();
                  runAiAnalysis();
                }}
                disabled={isRunningAnalysis}
              >
                {isRunningAnalysis ? (
                  <>
                    <span className="spinner-border spinner-border-sm me-1" role="status" aria-hidden="true"></span>
                    Updating...
                  </>
                ) : (
                  <>Run Analysis Now</>
                )}
              </button>
            </div>
          </div>

          <div className="card mt-3">
            <div className="card-body">
              <h5 className="card-title">Radar Controls</h5>
              <div className="form-check form-switch mb-3">
                <input 
                  className="form-check-input" 
                  type="checkbox" 
                  id="radarToggle" 
                  checked={showRadar} 
                  onChange={(e) => setShowRadar(e.target.checked)} 
                />
                <label className="form-check-label" htmlFor="radarToggle">
                  Show Radar
                </label>
              </div>
              <div className="mb-3">
                <label htmlFor="radarOpacity" className="form-label">Radar Opacity: {radarOpacity}</label>
                <input 
                  type="range" 
                  className="form-range" 
                  min="0.1" 
                  max="1" 
                  step="0.1" 
                  id="radarOpacity" 
                  value={radarOpacity}
                  onChange={(e) => setRadarOpacity(parseFloat(e.target.value))}
                />
              </div>
              <button 
                className="btn btn-sm btn-secondary mt-2" 
                onClick={fetchRadarData}
                disabled={isLoadingRadar}
              >
                {isLoadingRadar ? (
                  <>
                    <span className="spinner-border spinner-border-sm me-1" role="status" aria-hidden="true"></span>
                    Loading...
                  </>
                ) : (
                  <>Refresh Radar</>
                )}
              </button>
            </div>
          </div>
        </div>
        
        <div className="col-md-9">
          <div className="card">
            <div className="card-body p-0">
              <div style={{ height: '600px', width: '100%' }}>
                <MapContainer 
                  center={Array.isArray(mapCenter) && mapCenter.length === 2 && 
                          !isNaN(mapCenter[0]) && !isNaN(mapCenter[1]) ? 
                          mapCenter : [39.8283, -98.5795]}
                  zoom={mapZoom} 
                  style={{ height: '100%', width: '100%' }}
                  zoomControl={true}
                >
                  <MapUpdater center={Array.isArray(mapCenter) && mapCenter.length === 2 && 
                              !isNaN(mapCenter[0]) && !isNaN(mapCenter[1]) ? 
                              mapCenter : [39.8283, -98.5795]} 
                              zoom={mapZoom} />
                  
                  <LayersControl position="topright">
                    <LayersControl.BaseLayer checked name="OpenStreetMap">
                      <TileLayer
                        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                      />
                    </LayersControl.BaseLayer>
                    <LayersControl.BaseLayer name="Satellite">
                      <TileLayer
                        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                        attribution='&copy; <a href="https://www.esri.com/">Esri</a>'
                      />
                    </LayersControl.BaseLayer>
                    <LayersControl.BaseLayer name="Topographic">
                      <TileLayer
                        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
                        attribution='&copy; <a href="https://www.esri.com/">Esri</a>'
                      />
                    </LayersControl.BaseLayer>
                  </LayersControl>
                  
                  {/* Radar Overlay */}
                  <RadarOverlay 
                    isVisible={showRadar} 
                    opacity={radarOpacity} 
                    radarUrl={radarUrl} 
                    bounds={radarBounds} 
                  />
                  
                  {/* Render NWS Alerts */}
                  {nwsAlerts.filter(alert => isValidLatLng(alert.lat, alert.lon)).map((alert, index) => (
                    <Circle 
                      key={`alert-${alert.id || index}`}
                      center={[alert.lat, alert.lon]}
                      radius={alert.radius || 30000}
                      pathOptions={{
                        fillColor: getAlertSeverityColor(alert.severity),
                        fillOpacity: 0.2,
                        color: getAlertSeverityColor(alert.severity),
                        weight: 2,
                        dashArray: '5, 5'
                      }}
                    />
                  ))}
                  
                  {nwsAlerts.filter(alert => isValidLatLng(alert.lat, alert.lon)).map((alert, index) => (
                    <Marker 
                      key={`alert-marker-${alert.id || index}`}
                      position={[alert.lat, alert.lon]}
                      icon={alertIcon}
                    >
                      <Popup>
                        <div>
                          <h6>{alert.headline || 'Weather Alert'}</h6>
                          <p>
                            <strong>Type:</strong> {alert.event_type || 'Unknown'}<br />
                            <strong>Severity:</strong> {alert.severity || 'Unknown'}<br />
                            <strong>Issued:</strong> {new Date(alert.effective).toLocaleString() || 'Unknown'}<br />
                            <strong>Expires:</strong> {new Date(alert.expires).toLocaleString() || 'Unknown'}<br />
                          </p>
                          <div className="alert alert-warning p-2 mb-0 mt-2">
                            <small>{alert.description || 'No description available'}</small>
                          </div>
                        </div>
                      </Popup>
                    </Marker>
                  ))}
                  
                  {/* Render AI Predictions */}
                  {predictions.filter(prediction => isValidLatLng(prediction.lat, prediction.lon)).map((prediction, index) => (
                    <React.Fragment key={`prediction-${prediction.id || index}`}>
                      <Circle 
                        center={[prediction.lat, prediction.lon]}
                        radius={getCircleRadius(prediction.risk_level)}
                        pathOptions={{
                          fillColor: getRiskColor(prediction.risk_level),
                          fillOpacity: 0.3,
                          color: getRiskColor(prediction.risk_level),
                          weight: 1
                        }}
                      />
                      <Marker position={[prediction.lat, prediction.lon]}>
                        <Popup>
                          <div>
                            <h6>{prediction.location || 'Unknown Location'}</h6>
                            <p>
                              <strong>Risk Level:</strong> {prediction.risk_level || 'Unknown'}<br />
                              <strong>Formation Chance:</strong> {prediction.formation_chance || 0}%<br />
                              <strong>Prediction Time:</strong> {new Date(prediction.prediction_time).toLocaleString() || 'Unknown'}<br />
                              {prediction.cape && <><strong>CAPE:</strong> {prediction.cape} J/kg<br /></>}
                              {prediction.helicity && <><strong>Helicity:</strong> {prediction.helicity} m²/s²<br /></>}
                              {prediction.shear && <><strong>Wind Shear:</strong> {prediction.shear}<br /></>}
                            </p>
                            {prediction.nws_alert && (
                              <div className="alert alert-warning p-2 mb-0 mt-2">
                                <small><strong>NWS Alert Active</strong> for this area</small>
                              </div>
                            )}
                          </div>
                        </Popup>
                      </Marker>
                    </React.Fragment>
                  ))}
                </MapContainer>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <div className="row">
        <div className="col-12">
          <div className="card">
            <div className="card-body">
              <h5 className="card-title">Recent Predictions</h5>
              {isLoadingPredictions ? (
                <div className="d-flex justify-content-center my-3">
                  <div className="spinner-border text-primary" role="status">
                    <span className="visually-hidden">Loading...</span>
                  </div>
                </div>
              ) : (
                <div className="table-responsive">
                  <table className="table table-hover">
                    <thead>
                      <tr>
                        <th>Location</th>
                        <th>Risk Level</th>
                        <th>Formation Chance</th>
                        <th>CAPE</th>
                        <th>Helicity</th>
                        <th>Predicted At</th>
                      </tr>
                    </thead>
                    <tbody>
                      {predictions.slice(0, 5).map((prediction, index) => (
                        <tr key={`table-${prediction.id || index}`}>
                          <td>{prediction.location || 'Unknown'}</td>
                          <td>
                            <span className={`badge bg-${
                              prediction.risk_level === 'low' ? 'success' : 
                              prediction.risk_level === 'medium' ? 'warning' : 
                              prediction.risk_level === 'high' ? 'danger' : 
                              'dark'
                            }`}>
                              {prediction.risk_level || 'Unknown'}
                            </span>
                          </td>
                          <td>{prediction.formation_chance || 0}%</td>
                          <td>{prediction.cape || 'N/A'}</td>
                          <td>{prediction.helicity || 'N/A'}</td>
                          <td>{new Date(prediction.prediction_time).toLocaleString() || 'Unknown'}</td>
                        </tr>
                      ))}
                      {predictions.length === 0 && (
                        <tr>
                          <td colSpan="6" className="text-center">No predictions available</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TornadoMap; 
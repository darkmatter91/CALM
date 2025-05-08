/**
 * Tornado Model Integration
 * Connects the TornadoDataCollector and TornadoPredictionModel with the UI
 */

// Initialize components when the page loads
document.addEventListener('DOMContentLoaded', () => {
  initializeTornadoAIPrediction();
});

let dataCollector = null;
let predictionModel = null;
let modelInitialized = false;

/**
 * Initialize the tornado AI prediction system
 */
async function initializeTornadoAIPrediction() {
  try {
    // Create status message in UI
    showStatusMessage('Initializing AI prediction system...');
    
    // Load the required scripts if not already loaded
    await loadRequiredScripts();
    
    // Initialize the data collector
    dataCollector = new TornadoDataCollector();
    await dataCollector.initialize();
    
    // Initialize the prediction model
    predictionModel = new TornadoPredictionModel({
      trainingEnabled: true,
      minimumConfidenceThreshold: 0.15 // 15% minimum threshold
    });
    await predictionModel.initialize(dataCollector);
    
    modelInitialized = true;
    
    // Set up UI components
    setupTornadoPredictionUI();
    
    // Check for pending predictions to verify
    checkPendingPredictions();
    
    // Schedule regular checks for pending predictions
    setInterval(checkPendingPredictions, 60 * 60 * 1000); // Check every hour
    
    showStatusMessage('AI prediction system ready', 'success', 3000);
    console.log('Tornado AI prediction system initialized');
  } catch (error) {
    console.error('Failed to initialize tornado prediction system:', error);
    showStatusMessage('Failed to initialize AI prediction system', 'error');
  }
}

/**
 * Load required JavaScript files if not already loaded
 */
async function loadRequiredScripts() {
  const scriptsToLoad = [
    '/tornado_data_collector.js',
    '/tornado_prediction_model.js'
  ];
  
  const loadPromises = scriptsToLoad.map(scriptPath => {
    // Check if script is already loaded
    if (document.querySelector(`script[src="${scriptPath}"]`)) {
      return Promise.resolve();
    }
    
    // Load the script
    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = scriptPath;
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });
  });
  
  return Promise.all(loadPromises);
}

/**
 * Set up the UI components for tornado prediction
 */
function setupTornadoPredictionUI() {
  // Add AI prediction toggle to settings
  addAIPredictionToggle();
  
  // Add accuracy statistics panel
  addAccuracyStatsPanel();
  
  // Add event listeners
  addPredictionEventListeners();
}

/**
 * Add AI prediction toggle to settings
 */
function addAIPredictionToggle() {
  const settingsContainer = document.querySelector('.settings-container') || 
                           document.querySelector('#settings-panel');
  
  if (!settingsContainer) return;
  
  const toggleHtml = `
    <div class="settings-item ai-prediction-toggle">
      <label for="ai-prediction-toggle">AI Tornado Prediction</label>
      <div class="toggle-switch">
        <input type="checkbox" id="ai-prediction-toggle" checked>
        <span class="toggle-slider"></span>
      </div>
    </div>
  `;
  
  settingsContainer.insertAdjacentHTML('beforeend', toggleHtml);
  
  // Add event listener
  document.getElementById('ai-prediction-toggle').addEventListener('change', function(e) {
    const enabled = e.target.checked;
    toggleAIPredictions(enabled);
  });
}

/**
 * Add model accuracy statistics panel
 */
function addAccuracyStatsPanel() {
  // Find appropriate container
  const container = document.querySelector('.sidebar') || 
                   document.querySelector('#tornado-data-panel');
  
  if (!container) return;
  
  const statsHtml = `
    <div class="accuracy-stats-panel" style="display: none;">
      <h3>AI Model Training</h3>
      <div class="stats-container">
        <div class="stat-item">
          <span class="stat-label">Predictions Analyzed:</span>
          <span class="stat-value" id="total-predictions">0</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Model Accuracy:</span>
          <span class="stat-value" id="model-accuracy">0%</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Correct Predictions:</span>
          <span class="stat-value" id="correct-predictions">0</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">False Alarms:</span>
          <span class="stat-value" id="false-positives">0</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Missed Events:</span>
          <span class="stat-value" id="false-negatives">0</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Last Updated:</span>
          <span class="stat-value" id="last-updated">Never</span>
        </div>
      </div>
    </div>
  `;
  
  container.insertAdjacentHTML('beforeend', statsHtml);
  
  // Add a button to show/hide the stats panel
  const buttonHtml = `
    <button id="show-stats-button" class="btn btn-info btn-sm">
      <i class="fas fa-chart-bar"></i> AI Training Stats
    </button>
  `;
  
  const buttonContainer = document.querySelector('.map-controls') || 
                         document.querySelector('#tornado-controls');
  
  if (buttonContainer) {
    buttonContainer.insertAdjacentHTML('beforeend', buttonHtml);
    
    // Add event listener
    document.getElementById('show-stats-button').addEventListener('click', toggleStatsPanel);
  }
  
  // Update stats initially
  updateAccuracyStats();
}

/**
 * Toggle the visibility of the stats panel
 */
function toggleStatsPanel() {
  const panel = document.querySelector('.accuracy-stats-panel');
  if (panel) {
    const isVisible = panel.style.display !== 'none';
    panel.style.display = isVisible ? 'none' : 'block';
    
    // Update stats when showing
    if (!isVisible) {
      updateAccuracyStats();
    }
  }
}

/**
 * Update the accuracy statistics in the UI
 */
function updateAccuracyStats() {
  if (!predictionModel) return;
  
  const stats = predictionModel.getModelAccuracy();
  
  // Update UI elements
  document.getElementById('total-predictions').textContent = stats.totalPredictions;
  document.getElementById('model-accuracy').textContent = 
    (stats.accuracy * 100).toFixed(1) + '%';
  document.getElementById('correct-predictions').textContent = stats.correctPredictions;
  document.getElementById('false-positives').textContent = stats.falsePositives;
  document.getElementById('false-negatives').textContent = stats.falseNegatives;
  
  if (stats.lastUpdated) {
    const lastUpdated = new Date(stats.lastUpdated);
    document.getElementById('last-updated').textContent = 
      lastUpdated.toLocaleDateString() + ' ' + lastUpdated.toLocaleTimeString();
  }
}

/**
 * Add event listeners for tornado prediction functionality
 */
function addPredictionEventListeners() {
  // Listen for ZIP code submissions
  const zipForm = document.getElementById('zip-form');
  if (zipForm) {
    zipForm.addEventListener('submit', function(e) {
      // Note: don't prevent default as the original handler still needs to run
      const zipInput = document.getElementById('zip-input');
      if (zipInput && zipInput.value) {
        makePredictionForZipCode(zipInput.value);
      }
    });
  }
  
  // Listen for map clicks
  const map = window.tornadoMap || window.map;
  if (map) {
    map.on('click', function(e) {
      if (isAIPredictionEnabled()) {
        makePredictionForLocation({
          lat: e.latlng.lat,
          lon: e.latlng.lng
        });
      }
    });
  }
}

/**
 * Make a tornado prediction for a given ZIP code
 * @param {String} zipCode - ZIP code to predict for
 */
async function makePredictionForZipCode(zipCode) {
  if (!modelInitialized || !isAIPredictionEnabled()) return;
  
  try {
    // Get location data for ZIP code
    const locationData = await fetchLocationForZipCode(zipCode);
    if (!locationData) {
      console.error('Could not find location data for ZIP code', zipCode);
      return;
    }
    
    // Make prediction using the model
    await makePredictionForLocation(locationData);
  } catch (error) {
    console.error('Error making prediction for ZIP code:', error);
  }
}

/**
 * Make a tornado prediction for a given location
 * @param {Object} location - Location with lat/lon
 */
async function makePredictionForLocation(location) {
  if (!modelInitialized || !isAIPredictionEnabled()) return;
  
  try {
    showStatusMessage('Generating AI prediction...', 'info');
    
    // Enhance location data with nearby landmarks/cities
    const enhancedLocation = await enhanceLocationData(location);
    
    // Make the prediction
    const predictions = await predictionModel.makePredictions(enhancedLocation);
    
    // Display the predictions on the map
    displayPredictionsOnMap(predictions);
    
    if (predictions.length > 0) {
      showStatusMessage(
        `AI predicted ${predictions.length} potential tornado risk area${predictions.length > 1 ? 's' : ''}`, 
        'warning',
        5000
      );
    } else {
      showStatusMessage('No significant tornado risk predicted by AI', 'success', 3000);
    }
  } catch (error) {
    console.error('Error making prediction for location:', error);
    showStatusMessage('Error generating prediction', 'error');
  }
}

/**
 * Enhance location data with nearby landmarks and cities
 * @param {Object} location - Basic location with lat/lon
 * @returns {Object} Enhanced location data
 */
async function enhanceLocationData(location) {
  try {
    // Try to get county information
    const county = dataCollector.findCountyForLocation(location);
    
    return {
      ...location,
      county: county ? county.name : null,
      state: county ? county.state : null
    };
  } catch (error) {
    console.warn('Error enhancing location data:', error);
    return location;
  }
}

/**
 * Display tornado predictions on the map
 * @param {Array} predictions - Array of tornado predictions
 */
function displayPredictionsOnMap(predictions) {
  const map = window.tornadoMap || window.map;
  if (!map) return;
  
  // Check if we have a layer group for AI predictions
  if (!window.aiPredictionLayer) {
    window.aiPredictionLayer = L.layerGroup().addTo(map);
  }
  
  // Clear previous predictions
  window.aiPredictionLayer.clearLayers();
  
  // Add each prediction to the map
  predictions.forEach(prediction => {
    // Create markers and paths for the prediction
    addPredictionToMap(prediction, window.aiPredictionLayer);
    
    // Zoom to prediction if it's high confidence
    if (prediction.formationChance > 60) {
      map.setView([prediction.lat, prediction.lon], 9);
    }
  });
}

/**
 * Check if AI prediction is enabled
 * @returns {Boolean} True if enabled
 */
function isAIPredictionEnabled() {
  const toggle = document.getElementById('ai-prediction-toggle');
  return toggle ? toggle.checked : true;
}

/**
 * Toggle AI prediction functionality
 * @param {Boolean} enabled - Whether AI prediction should be enabled
 */
function toggleAIPredictions(enabled) {
  // Clear AI predictions if disabled
  if (!enabled && window.aiPredictionLayer) {
    window.aiPredictionLayer.clearLayers();
  }
  
  // Show/hide AI-related UI elements
  const aiElements = document.querySelectorAll('.ai-prediction-related');
  aiElements.forEach(el => {
    el.style.display = enabled ? '' : 'none';
  });
  
  showStatusMessage(
    enabled ? 'AI predictions enabled' : 'AI predictions disabled',
    'info',
    2000
  );
}

/**
 * Check for pending predictions to verify
 */
async function checkPendingPredictions() {
  if (!modelInitialized) return;
  
  try {
    const count = await predictionModel.checkPendingPredictions();
    
    if (count > 0) {
      console.log(`Verified ${count} pending predictions`);
      updateAccuracyStats();
    }
  } catch (error) {
    console.error('Error checking pending predictions:', error);
  }
}

/**
 * Fetch location data for a ZIP code
 * @param {String} zipCode - ZIP code to look up
 * @returns {Object} Location data with lat/lon
 */
async function fetchLocationForZipCode(zipCode) {
  // Try to get from data collector first
  if (dataCollector) {
    const county = await dataCollector.getCountyForZipCode(zipCode);
    if (county && county.lat && county.lon) {
      return {
        lat: county.lat,
        lon: county.lon,
        zipCode,
        county: county.name,
        state: county.state
      };
    }
  }
  
  // Fall back to existing functionality if available
  if (typeof getCoordinatesForZipCode === 'function') {
    return getCoordinatesForZipCode(zipCode);
  }
  
  // Hard-coded fallback for common ZIP codes
  const zipMapping = {
    '90210': { lat: 34.1030, lon: -118.4105, city: 'Beverly Hills', state: 'CA' },
    '10001': { lat: 40.7501, lon: -73.9997, city: 'New York', state: 'NY' },
    '73301': { lat: 30.2672, lon: -97.7431, city: 'Austin', state: 'TX' },
    '60601': { lat: 41.8842, lon: -87.6209, city: 'Chicago', state: 'IL' }
  };
  
  return zipMapping[zipCode] || null;
}

/**
 * Show a status message in the UI
 * @param {String} message - Message to display
 * @param {String} type - Message type (info, success, warning, error)
 * @param {Number} duration - How long to show the message (ms)
 */
function showStatusMessage(message, type = 'info', duration = 0) {
  // Find or create status container
  let statusContainer = document.getElementById('status-messages');
  
  if (!statusContainer) {
    statusContainer = document.createElement('div');
    statusContainer.id = 'status-messages';
    statusContainer.style.position = 'fixed';
    statusContainer.style.bottom = '20px';
    statusContainer.style.right = '20px';
    statusContainer.style.zIndex = '1000';
    document.body.appendChild(statusContainer);
  }
  
  // Create message element
  const msgElement = document.createElement('div');
  msgElement.className = `status-message ${type}`;
  msgElement.textContent = message;
  
  // Style the message
  msgElement.style.padding = '10px 15px';
  msgElement.style.marginTop = '10px';
  msgElement.style.borderRadius = '4px';
  msgElement.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
  msgElement.style.backgroundColor = 
    type === 'success' ? '#4CAF50' :
    type === 'warning' ? '#FF9800' :
    type === 'error' ? '#F44336' : '#2196F3';
  msgElement.style.color = 'white';
  msgElement.style.transition = 'opacity 0.5s';
  
  // Add close button
  const closeBtn = document.createElement('span');
  closeBtn.textContent = '×';
  closeBtn.style.marginLeft = '10px';
  closeBtn.style.cursor = 'pointer';
  closeBtn.style.float = 'right';
  closeBtn.onclick = function() {
    statusContainer.removeChild(msgElement);
  };
  msgElement.appendChild(closeBtn);
  
  // Add to container
  statusContainer.appendChild(msgElement);
  
  // Remove after duration if specified
  if (duration > 0) {
    setTimeout(() => {
      if (statusContainer.contains(msgElement)) {
        // Fade out
        msgElement.style.opacity = '0';
        setTimeout(() => {
          if (statusContainer.contains(msgElement)) {
            statusContainer.removeChild(msgElement);
          }
        }, 500);
      }
    }, duration);
  }
}

/**
 * Add a tornado prediction to the map
 * @param {Object} prediction - Tornado prediction data
 * @param {L.LayerGroup} layerGroup - Layer group to add to
 */
function addPredictionToMap(prediction, layerGroup) {
  // Skip if the map or layer group is not available
  if (!layerGroup) return;
  
  // Calculate the EF color based on predicted magnitude
  const getEFColor = (magnitude) => {
    const colors = [
      '#92d050', // EF0: Light green
      '#ffff00', // EF1: Yellow
      '#ffc000', // EF2: Orange
      '#ff0000', // EF3: Red
      '#7030a0', // EF4: Purple
      '#000000'  // EF5: Black
    ];
    return colors[Math.min(5, magnitude)];
  };
  
  const efColor = getEFColor(prediction.magnitude);
  
  // Create a marker for the predicted formation point
  const marker = L.circleMarker([prediction.lat, prediction.lon], {
    radius: 8,
    color: 'black',
    weight: 2,
    fillColor: efColor,
    fillOpacity: 0.7
  }).addTo(layerGroup);
  
  // Add a popup with information
  const popupContent = `
    <div class="prediction-popup">
      <h4>AI Tornado Prediction</h4>
      <p><strong>Location:</strong> ${prediction.county || ''}, ${prediction.state || ''}</p>
      <p><strong>Formation Chance:</strong> ${prediction.formationChance}%</p>
      <p><strong>Predicted Magnitude:</strong> EF${prediction.magnitude}</p>
      <p><strong>Predicted Time:</strong> ${formatTime(prediction.estimatedFormationTime)}</p>
      <p><strong>Est. Duration:</strong> ${prediction.estimatedDuration} minutes</p>
      <p class="ai-disclaimer">This is an AI prediction and should be verified with official NWS alerts.</p>
    </div>
  `;
  marker.bindPopup(popupContent);
  
  // Add a warning circle based on magnitude and formation chance
  const warningRadius = (prediction.magnitude + 1) * 5000 * (prediction.formationChance / 50);
  L.circle([prediction.lat, prediction.lon], {
    radius: warningRadius,
    color: efColor,
    fillColor: efColor,
    fillOpacity: 0.15,
    weight: 1
  }).addTo(layerGroup);
  
  // If we have a predicted path, add it to the map
  if (prediction.predictedPath && prediction.predictedPath.length > 1) {
    // Extract coordinates for the path
    const pathCoords = prediction.predictedPath.map(point => [point.lat, point.lon]);
    
    // Create a polyline for the path
    const pathLine = L.polyline(pathCoords, {
      color: efColor,
      weight: 3,
      opacity: 0.7,
      dashArray: '5, 5',
      smoothFactor: 1
    }).addTo(layerGroup);
    
    // Add time markers along the path
    prediction.predictedPath.forEach((point, index) => {
      // Only add markers for a few points
      if (index > 0 && (index % 2 === 0 || index === prediction.predictedPath.length - 1)) {
        const timeMarker = L.circleMarker([point.lat, point.lon], {
          radius: 3,
          color: efColor,
          fillColor: '#fff',
          fillOpacity: 0.8,
          weight: 2
        }).addTo(layerGroup);
        
        timeMarker.bindTooltip(formatTime(point.time), {
          permanent: false,
          direction: 'top'
        });
      }
    });
  }
  
  // Add "AI Prediction" label
  const aiLabel = L.divIcon({
    className: 'ai-prediction-label',
    html: 'AI Prediction',
    iconSize: [80, 20],
    iconAnchor: [40, 25]
  });
  
  L.marker([prediction.lat, prediction.lon], {
    icon: aiLabel,
    interactive: false
  }).addTo(layerGroup);
}

/**
 * Format a time string nicely
 * @param {String} timeString - ISO time string
 * @returns {String} Formatted time
 */
function formatTime(timeString) {
  try {
    const date = new Date(timeString);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch (e) {
    return timeString;
  }
} 
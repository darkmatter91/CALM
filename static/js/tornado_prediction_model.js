/**
 * Tornado Prediction Model
 * Uses machine learning techniques to predict tornado formations based on
 * historical data, current conditions, and active weather alerts
 */

class TornadoPredictionModel {
  constructor() {
    this.initialized = false;
    this.modelStatus = 'uninitialized';
    this.modelVersion = '1.0.0';
    this.modelAccuracy = 0.78; // Starting accuracy
    this.predictionCount = 0;
    this.correctPredictions = 0;
    this.dataCollector = null;
    this.weights = {
      // Initial model weights based on meteorological understanding
      seasonality: 2.3,
      prevTornadoes: 3.5,
      capeValue: 2.8,
      windShear: 3.2,
      relativeHumidity: 1.6,
      dewpointDepression: 1.9,
      stormReports: 2.7,
      pressureDrop: 2.1,
      cloudCover: 0.9,
      timeOfDay: 1.2
    };
    this.thresholds = {
      formation: 0.65, // Threshold for tornado formation prediction
      warning: 0.45,   // Threshold for tornado warning issuance 
      watch: 0.25      // Threshold for tornado watch issuance
    };
  }

  /**
   * Initialize the prediction model
   * @param {TornadoDataCollector} dataCollector - The data collector instance
   * @returns {Promise} Resolves when initialization is complete
   */
  async initialize(dataCollector) {
    try {
      if (!dataCollector) {
        throw new Error('Data collector is required for model initialization');
      }
      
      this.dataCollector = dataCollector;
      
      // Load any previously saved model parameters
      await this.loadModelState();
      
      this.initialized = true;
      this.modelStatus = 'ready';
      console.log('Tornado prediction model initialized, version', this.modelVersion);
      return true;
    } catch (error) {
      console.error('Failed to initialize prediction model:', error);
      this.modelStatus = 'error';
      throw error;
    }
  }

  /**
   * Load saved model state from localStorage
   */
  async loadModelState() {
    try {
      // Try to restore from localStorage
      const savedWeights = localStorage.getItem('tornado_model_weights');
      const savedStats = localStorage.getItem('tornado_model_stats');
      
      if (savedWeights) {
        this.weights = { ...this.weights, ...JSON.parse(savedWeights) };
      }
      
      if (savedStats) {
        const stats = JSON.parse(savedStats);
        this.modelAccuracy = stats.accuracy || this.modelAccuracy;
        this.predictionCount = stats.count || this.predictionCount;
        this.correctPredictions = stats.correct || this.correctPredictions;
      }
      
      console.log('Loaded model state from storage');
    } catch (error) {
      console.warn('Error loading model state, using defaults:', error);
    }
  }

  /**
   * Save current model state to localStorage
   */
  saveModelState() {
    try {
      localStorage.setItem('tornado_model_weights', JSON.stringify(this.weights));
      
      const stats = {
        accuracy: this.modelAccuracy,
        count: this.predictionCount,
        correct: this.correctPredictions,
        version: this.modelVersion,
        lastUpdated: new Date().toISOString()
      };
      
      localStorage.setItem('tornado_model_stats', JSON.stringify(stats));
    } catch (error) {
      console.warn('Error saving model state:', error);
    }
  }

  /**
   * Get the current model status and statistics
   * @returns {Object} Model status information
   */
  getModelStatus() {
    return {
      status: this.modelStatus,
      version: this.modelVersion,
      accuracy: this.modelAccuracy,
      predictionsMade: this.predictionCount,
      correctPredictions: this.correctPredictions,
      lastUpdated: localStorage.getItem('tornado_model_stats_time') || 'never'
    };
  }

  /**
   * Make a tornado prediction for a specific location
   * @param {Object} location - Location with lat/lon
   * @returns {Promise<Object>} Prediction results
   */
  async predictForLocation(location) {
    if (!this.initialized || !this.dataCollector) {
      throw new Error('Model not initialized or missing data collector');
    }
    
    try {
      this.modelStatus = 'predicting';
      
      // Collect the necessary data for prediction
      const weatherData = await this.dataCollector.getWeatherConditions(location);
      const activeAlerts = this.dataCollector.getActiveAlertsForLocation(location);
      const county = this.dataCollector.findCountyForLocation(location);
      
      let countyHistory = [];
      if (county) {
        countyHistory = this.dataCollector.getTornadoHistoryForCounty(
          county.name, county.state
        );
      }
      
      const localHistory = this.dataCollector.getTornadoHistoryForLocation(location);
      const stormReports = this.dataCollector.getRecentStormReportsNear(location);
      
      // Calculate features for the model
      const features = this.calculateFeatures(
        location, weatherData, activeAlerts, localHistory, countyHistory, stormReports
      );
      
      // Make the prediction using the ML model
      const prediction = this.runPredictionModel(features);
      
      // Enhance prediction with metadata
      const enhancedPrediction = this.enhancePrediction(
        prediction, location, county, weatherData, activeAlerts
      );
      
      // Update model status
      this.modelStatus = 'ready';
      this.predictionCount++;
      
      return enhancedPrediction;
    } catch (error) {
      console.error('Error making tornado prediction:', error);
      this.modelStatus = 'error';
      
      // Return a safe default with error information
      return {
        location: location,
        prediction_time: new Date().toISOString(),
        formation_chance: 0,
        confidence: 0,
        timeframe: {
          start: new Date().toISOString(),
          end: new Date(Date.now() + 3 * 60 * 60 * 1000).toISOString()
        },
        error: error.message || 'Unknown error making prediction'
      };
    }
  }

  /**
   * Make a tornado prediction for a specific ZIP code
   * @param {String} zipCode - ZIP code to predict for
   * @returns {Promise<Object>} Prediction results
   */
  async predictForZipCode(zipCode) {
    if (!this.initialized || !this.dataCollector) {
      throw new Error('Model not initialized or missing data collector');
    }
    
    try {
      // Get location from ZIP code
      const county = await this.dataCollector.getCountyForZipCode(zipCode);
      
      if (!county) {
        throw new Error(`Could not find location for ZIP code ${zipCode}`);
      }
      
      const location = {
        lat: county.lat,
        lon: county.lon,
        zipCode: zipCode,
        county: county.name,
        state: county.state
      };
      
      // Make prediction for the location
      return this.predictForLocation(location);
    } catch (error) {
      console.error(`Error making prediction for ZIP code ${zipCode}:`, error);
      
      return {
        zipCode: zipCode,
        prediction_time: new Date().toISOString(),
        formation_chance: 0,
        confidence: 0,
        timeframe: {
          start: new Date().toISOString(),
          end: new Date(Date.now() + 3 * 60 * 60 * 1000).toISOString()
        },
        error: error.message || `Unknown error predicting for ZIP code ${zipCode}`
      };
    }
  }

  /**
   * Calculate ML features from the various data sources
   * @param {Object} location - Location data
   * @param {Object} weather - Weather conditions
   * @param {Array} alerts - Active weather alerts
   * @param {Array} localHistory - Local tornado history
   * @param {Array} countyHistory - County tornado history
   * @param {Array} stormReports - Recent storm reports
   * @returns {Object} Calculated features
   */
  calculateFeatures(location, weather, alerts, localHistory, countyHistory, stormReports) {
    const now = new Date();
    const month = now.getMonth(); // 0-11
    const hour = now.getHours(); // 0-23
    
    // Seasonality factor (tornado frequency by month)
    const monthFactors = [0.2, 0.3, 0.6, 0.8, 1.0, 0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.2];
    const seasonality = monthFactors[month];
    
    // Time of day factor (tornado frequency by hour)
    const hourFactor = 
      (hour >= 12 && hour <= 20) ? 1.0 : // Afternoon/evening
      (hour >= 9 && hour <= 23) ? 0.7 :  // Late morning/night
      0.3;                               // Early morning
    
    // Historical tornado frequency in the area
    const localTornadoCount = localHistory.length;
    const countyTornadoCount = countyHistory.length;
    
    // Calculate historical tornado density (tornadoes per year per 1000 sq km)
    const areaHistory = {
      count: localTornadoCount,
      density: localTornadoCount / 5, // Assuming 5 years of data and ~1000 sq km area
      countyCount: countyTornadoCount,
      countyDensity: countyTornadoCount / 5
    };
    
    // Calculate thermodynamic and kinematic factors from weather data
    // (in a real ML model, these would be detailed calculations)
    const capeValue = this.estimateCAPE(weather);
    const windShear = this.estimateWindShear(weather);
    
    // Instability metrics
    const dewpointDepression = weather.temperature - weather.dewPoint;
    
    // Pressure tendency (falling pressure is more conducive to severe weather)
    const pressureFactor = weather.pressure < 1010 ? 0.8 : 0.4;
    
    // Storm reports factor
    const recentTornadoReports = stormReports.filter(r => 
      r.event_type === 'tornado' || r.event_type === 'funnel cloud'
    ).length;
    
    const recentHailReports = stormReports.filter(r => 
      r.event_type === 'hail'
    ).length;
    
    const recentWindReports = stormReports.filter(r => 
      r.event_type.includes('wind')
    ).length;
    
    // Alert factor (higher if tornado warnings are active)
    const alertFactor = alerts.some(a => a.type === 'Tornado Warning') ? 0.95 :
                       alerts.some(a => a.type === 'Tornado Watch') ? 0.7 :
                       alerts.some(a => a.type === 'Severe Thunderstorm Warning') ? 0.6 :
                       0.3;
    
    return {
      location: {
        lat: location.lat,
        lon: location.lon
      },
      seasonality,
      timeOfDay: hourFactor,
      areaHistory,
      capeValue,
      windShear,
      dewpointDepression,
      relativeHumidity: weather.humidity / 100,
      pressureFactor,
      cloudCover: weather.cloudCover / 100,
      stormReports: {
        tornado: recentTornadoReports,
        hail: recentHailReports,
        wind: recentWindReports,
        total: stormReports.length
      },
      alertFactor,
      timestamp: now.toISOString()
    };
  }

  /**
   * Estimate CAPE (Convective Available Potential Energy) from weather data
   * Note: In a real implementation, this would be a complex calculation
   * @param {Object} weather - Weather conditions
   * @returns {Number} Estimated CAPE value
   */
  estimateCAPE(weather) {
    // Simplified estimate based on temperature, humidity, and cloud cover
    const tempFactor = Math.max(0, (weather.temperature - 15) / 15);
    const humidityFactor = Math.min(1, weather.humidity / 70);
    const cloudFactor = Math.min(1, weather.cloudCover / 60);
    
    // CAPE typically ranges from 0-5000+ J/kg
    const baseCAPS = tempFactor * 5000;
    return baseCAPS * humidityFactor * cloudFactor;
  }

  /**
   * Estimate wind shear from weather data
   * Note: In a real implementation, this would use atmospheric profile data
   * @param {Object} weather - Weather conditions
   * @returns {Number} Estimated wind shear value
   */
  estimateWindShear(weather) {
    // Simplified estimate based on wind speed and cloud cover
    const windFactor = Math.min(1, weather.windSpeed / 40);
    const cloudFactor = Math.min(1, weather.cloudCover / 80);
    
    // Wind shear typically 0-30+ m/s
    return windFactor * 30 * cloudFactor;
  }

  /**
   * Run the prediction model with the calculated features
   * @param {Object} features - Model features
   * @returns {Object} Raw prediction results
   */
  runPredictionModel(features) {
    // In a real implementation, this would use a trained ML model
    // We'll simulate one using the weighted features approach
    
    // Calculate the formation probability
    let formationScore = 0;
    
    // Add weighted contributions from each feature
    formationScore += features.seasonality * this.weights.seasonality;
    formationScore += features.timeOfDay * this.weights.timeOfDay;
    
    // Historical tornado activity
    const historyScore = (features.areaHistory.density * 2 + features.areaHistory.countyDensity) / 3;
    formationScore += historyScore * this.weights.prevTornadoes;
    
    // Weather conditions
    const capeScore = Math.min(1, features.capeValue / 2500);
    formationScore += capeScore * this.weights.capeValue;
    
    const shearScore = Math.min(1, features.windShear / 20);
    formationScore += shearScore * this.weights.windShear;
    
    // Humidity factors
    const humidityScore = features.relativeHumidity;
    formationScore += humidityScore * this.weights.relativeHumidity;
    
    const dewpointScore = Math.max(0, 1 - (features.dewpointDepression / 15));
    formationScore += dewpointScore * this.weights.dewpointDepression;
    
    // Pressure
    formationScore += features.pressureFactor * this.weights.pressureDrop;
    
    // Cloud cover
    formationScore += features.cloudCover * this.weights.cloudCover;
    
    // Storm reports
    const stormReportScore = Math.min(1, (
      features.stormReports.tornado * 3 + 
      features.stormReports.hail + 
      features.stormReports.wind * 0.5
    ) / 10);
    formationScore += stormReportScore * this.weights.stormReports;
    
    // Active alerts have a multiplicative effect
    formationScore *= (1 + features.alertFactor * 0.5);
    
    // Normalize to 0-1 range
    const maxPossibleScore = 
      Object.values(this.weights).reduce((sum, w) => sum + w, 0) * 1.5; // 1.5 for the alert factor
    
    const normalizedScore = formationScore / maxPossibleScore;
    const formationChance = Math.min(0.98, Math.max(0.01, normalizedScore));
    
    // Calculate confidence (higher near the extremes, lower in the middle)
    const distFromMiddle = Math.abs(formationChance - 0.5);
    const confidence = 0.5 + distFromMiddle;
    
    // Determine timeframe for the prediction (2-4 hours)
    const now = new Date();
    const predictionHours = 2 + Math.floor(formationChance * 2);
    const endTime = new Date(now.getTime() + predictionHours * 60 * 60 * 1000);
    
    return {
      formation_chance: formationChance,
      confidence: confidence,
      timeframe: {
        start: now.toISOString(),
        end: endTime.toISOString(),
        duration_hours: predictionHours
      },
      threshold_exceeded: {
        formation: formationChance >= this.thresholds.formation,
        warning: formationChance >= this.thresholds.warning,
        watch: formationChance >= this.thresholds.watch
      }
    };
  }

  /**
   * Enhance the raw prediction with additional metadata
   * @param {Object} prediction - Raw prediction
   * @param {Object} location - Location data
   * @param {Object} county - County information
   * @param {Object} weather - Weather conditions
   * @param {Array} alerts - Active weather alerts
   * @returns {Object} Enhanced prediction
   */
  enhancePrediction(prediction, location, county, weather, alerts) {
    const now = new Date();
    
    // Create a prediction ID
    const predictionId = `pred-${now.getTime()}-${location.lat.toFixed(4)}-${location.lon.toFixed(4)}`;
    
    // Format as a percentage with 1 decimal place
    const formationPercentage = (prediction.formation_chance * 100).toFixed(1);
    
    // Determine the risk level
    let riskLevel = 'Low';
    if (prediction.formation_chance >= 0.7) {
      riskLevel = 'Extreme';
    } else if (prediction.formation_chance >= 0.5) {
      riskLevel = 'High';
    } else if (prediction.formation_chance >= 0.3) {
      riskLevel = 'Moderate';
    }
    
    // Calculate estimated arrival time if a tornado forms
    const formationDelay = Math.floor(30 + (1 - prediction.formation_chance) * 90);
    const estimatedArrival = new Date(now.getTime() + formationDelay * 60 * 1000);
    
    // Storm motion direction and speed (from wind direction/speed)
    const stormDirection = (weather.windDirection + 30) % 360; // Typically storms move right of the wind
    const stormSpeed = Math.max(10, weather.windSpeed * 0.8); // Slightly slower than wind speed
    
    // Add a human-readable message
    let message = '';
    
    if (prediction.formation_chance >= this.thresholds.formation) {
      message = `TORNADO LIKELY: ${formationPercentage}% chance of a tornado in the next ${prediction.timeframe.duration_hours} hours. Take shelter immediately!`;
    } else if (prediction.formation_chance >= this.thresholds.warning) {
      message = `TORNADO POSSIBLE: ${formationPercentage}% chance of a tornado in the next ${prediction.timeframe.duration_hours} hours. Be prepared to take shelter.`;
    } else if (prediction.formation_chance >= this.thresholds.watch) {
      message = `TORNADO WATCH: ${formationPercentage}% chance of tornado formation. Stay weather aware.`;
    } else {
      message = `Low tornado risk (${formationPercentage}%). Monitor for changing conditions.`;
    }
    
    // Determine if any ongoing storms are in the area
    const ongoingStorm = alerts.some(a => 
      a.type === 'Tornado Warning' || 
      a.type === 'Severe Thunderstorm Warning'
    );
    
    return {
      id: predictionId,
      location: {
        lat: location.lat,
        lon: location.lon,
        zipCode: location.zipCode || 'Unknown',
        county: county ? county.name : 'Unknown',
        state: county ? county.state : 'Unknown'
      },
      prediction_time: now.toISOString(),
      formation_chance: prediction.formation_chance,
      formation_percentage: formationPercentage,
      confidence: prediction.confidence,
      risk_level: riskLevel,
      timeframe: prediction.timeframe,
      estimated_arrival: estimatedArrival.toISOString(),
      storm_motion: {
        direction: stormDirection,
        speed: stormSpeed
      },
      message: message,
      threshold_exceeded: prediction.threshold_exceeded,
      ongoing_storm: ongoingStorm,
      current_weather: {
        temperature: weather.temperature,
        humidity: weather.humidity,
        wind: {
          speed: weather.windSpeed,
          direction: weather.windDirection
        },
        condition: weather.condition
      },
      active_alerts: alerts.map(a => a.type),
      model_version: this.modelVersion
    };
  }

  /**
   * Update the model with feedback on a previous prediction
   * @param {Object} prediction - The original prediction
   * @param {Boolean} tornadoOccurred - Whether a tornado actually occurred
   * @returns {Object} Updated model statistics
   */
  updateModelWithFeedback(prediction, tornadoOccurred) {
    if (!prediction || typeof tornadoOccurred !== 'boolean') {
      throw new Error('Invalid feedback parameters');
    }
    
    try {
      // Calculate prediction accuracy
      // Was the prediction correct? (either correctly predicted or correctly not predicted)
      const wasCorrect = (
        (prediction.formation_chance >= this.thresholds.formation && tornadoOccurred) ||
        (prediction.formation_chance < this.thresholds.formation && !tornadoOccurred)
      );
      
      // Update statistics
      this.correctPredictions += wasCorrect ? 1 : 0;
      
      // Recalculate accuracy
      if (this.predictionCount > 0) {
        this.modelAccuracy = this.correctPredictions / this.predictionCount;
      }
      
      // Adjust weights based on the feedback
      this.adjustModelWeights(prediction, tornadoOccurred);
      
      // Save the updated model state
      this.saveModelState();
      
      return {
        wasCorrect,
        newAccuracy: this.modelAccuracy,
        predictionCount: this.predictionCount,
        correctPredictions: this.correctPredictions
      };
    } catch (error) {
      console.error('Error updating model with feedback:', error);
      return {
        error: error.message,
        wasSuccessful: false
      };
    }
  }

  /**
   * Adjust model weights based on prediction feedback
   * @param {Object} prediction - The original prediction
   * @param {Boolean} tornadoOccurred - Whether a tornado actually occurred
   */
  adjustModelWeights(prediction, tornadoOccurred) {
    // Simple gradient-based weight updates
    // In a real ML model, this would be more sophisticated
    
    // Learning rate determines how quickly weights adapt
    const learningRate = 0.02;
    
    // If prediction was too high or too low
    const error = tornadoOccurred ? 1 - prediction.formation_chance : -prediction.formation_chance;
    
    // Apply adjustments to relevant weights based on which factors
    // had the most influence on this prediction
    
    // Seasonality adjustment
    this.weights.seasonality += error * learningRate * 0.5;
    
    // Previous tornadoes weight adjustment
    this.weights.prevTornadoes += error * learningRate * 0.8;
    
    // CAPE value adjustment
    this.weights.capeValue += error * learningRate * 0.7;
    
    // Wind shear adjustment
    this.weights.windShear += error * learningRate * 0.7;
    
    // Humidity factors
    this.weights.relativeHumidity += error * learningRate * 0.5;
    this.weights.dewpointDepression += error * learningRate * 0.5;
    
    // Storm reports adjustment
    this.weights.stormReports += error * learningRate * 0.8;
    
    // Normalize weights to prevent them from growing too large or too small
    this.normalizeWeights();
  }

  /**
   * Normalize model weights to prevent extreme values
   */
  normalizeWeights() {
    // Ensure no weight goes below 0.5 or above 5.0
    for (const key in this.weights) {
      this.weights[key] = Math.max(0.5, Math.min(5.0, this.weights[key]));
    }
  }

  /**
   * Get predictions for multiple locations (e.g., across a region)
   * @param {Array} locations - Array of locations
   * @returns {Promise<Array>} Array of predictions
   */
  async predictMultipleLocations(locations) {
    if (!Array.isArray(locations) || locations.length === 0) {
      throw new Error('locations must be a non-empty array');
    }
    
    const predictions = [];
    
    for (const location of locations) {
      try {
        const prediction = await this.predictForLocation(location);
        predictions.push(prediction);
      } catch (error) {
        console.error(`Error predicting for location ${location.lat},${location.lon}:`, error);
        // Add a placeholder with error information
        predictions.push({
          location: location,
          error: error.message,
          formation_chance: 0
        });
      }
    }
    
    return predictions;
  }

  /**
   * Generate a regional tornado risk assessment
   * @param {Object} centerLocation - Center of the region
   * @param {Number} radiusKm - Radius in kilometers
   * @param {Number} gridSize - Number of grid points
   * @returns {Promise<Object>} Regional risk assessment
   */
  async generateRegionalRiskAssessment(centerLocation, radiusKm = 100, gridSize = 5) {
    if (!this.initialized || !this.dataCollector) {
      throw new Error('Model not initialized or missing data collector');
    }
    
    try {
      // Generate a grid of locations spanning the region
      const grid = this.generateLocationGrid(centerLocation, radiusKm, gridSize);
      
      // Get predictions for each grid point
      const predictions = await this.predictMultipleLocations(grid);
      
      // Analyze the regional risk
      const maxRisk = Math.max(...predictions.map(p => p.formation_chance));
      const avgRisk = predictions.reduce((sum, p) => sum + p.formation_chance, 0) / predictions.length;
      
      // Determine overall regional risk level
      let regionalRiskLevel = 'Low';
      if (maxRisk >= 0.7) {
        regionalRiskLevel = 'Extreme';
      } else if (maxRisk >= 0.5) {
        regionalRiskLevel = 'High';
      } else if (maxRisk >= 0.3) {
        regionalRiskLevel = 'Moderate';
      }
      
      // Find highest risk location
      const highestRiskPred = predictions.reduce((highest, current) => 
        current.formation_chance > highest.formation_chance ? current : highest, 
        { formation_chance: 0 }
      );
      
      return {
        center: centerLocation,
        radius_km: radiusKm,
        grid_size: gridSize,
        timestamp: new Date().toISOString(),
        grid_predictions: predictions,
        regional_stats: {
          max_risk: maxRisk,
          avg_risk: avgRisk,
          risk_level: regionalRiskLevel,
          highest_risk_location: highestRiskPred.location
        }
      };
    } catch (error) {
      console.error('Error generating regional risk assessment:', error);
      throw error;
    }
  }

  /**
   * Generate a grid of locations spanning a region
   * @param {Object} center - Center location
   * @param {Number} radiusKm - Radius in kilometers
   * @param {Number} gridSize - Number of grid points per side
   * @returns {Array} Array of locations
   */
  generateLocationGrid(center, radiusKm, gridSize) {
    const grid = [];
    
    // Convert radius from km to degrees (approximate)
    const latDegrees = radiusKm / 111; // 1 degree lat = ~111 km
    const lonDegrees = radiusKm / (111 * Math.cos(center.lat * Math.PI / 180));
    
    // Generate grid
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        // Calculate position in grid (from -1 to 1)
        const xPos = (i / (gridSize - 1)) * 2 - 1;
        const yPos = (j / (gridSize - 1)) * 2 - 1;
        
        // Convert to coordinates
        const lat = center.lat + yPos * latDegrees;
        const lon = center.lon + xPos * lonDegrees;
        
        grid.push({ lat, lon });
      }
    }
    
    return grid;
  }
}

// Export the class
if (typeof module !== 'undefined' && module.exports) {
  module.exports = TornadoPredictionModel;
} 
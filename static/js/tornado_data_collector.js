/**
 * Tornado Data Collector
 * Gathers historical tornado data, current weather conditions, and active alerts
 * to provide inputs for the tornado prediction model
 */

class TornadoDataCollector {
  constructor() {
    this.initialized = false;
    this.historicalData = [];
    this.stormReports = [];
    this.countyData = null;
    this.stateData = null;
    this.activeAlerts = [];
    this.currentWeatherConditions = {};
    this.lastUpdated = {
      historical: null,
      reports: null,
      alerts: null,
      weather: null
    };
  }

  /**
   * Initialize the data collector
   * @returns {Promise} Resolves when initialization is complete
   */
  async initialize() {
    try {
      // Load county and state boundary data
      await this.loadGeographicData();
      
      // Load historical tornado data
      await this.loadHistoricalTornadoData();
      
      // Load recent storm reports
      await this.loadRecentStormReports();
      
      // Set up alert monitoring
      this.setupAlertMonitoring();
      
      this.initialized = true;
      console.log('Tornado data collector initialized');
      return true;
    } catch (error) {
      console.error('Failed to initialize data collector:', error);
      throw error;
    }
  }

  /**
   * Load geographic data for counties and states
   */
  async loadGeographicData() {
    try {
      // Try to load from localStorage first
      const cachedCountyData = localStorage.getItem('county_data');
      
      if (cachedCountyData) {
        this.countyData = JSON.parse(cachedCountyData);
        console.log('Loaded county data from cache');
      } else {
        // Fallback to static sample data (simplified)
        this.countyData = this.getSampleCountyData();
        // Cache for future use
        localStorage.setItem('county_data', JSON.stringify(this.countyData));
      }
      
      // Load state boundaries similarly
      const cachedStateData = localStorage.getItem('state_data');
      
      if (cachedStateData) {
        this.stateData = JSON.parse(cachedStateData);
      } else {
        this.stateData = this.getSampleStateData();
        localStorage.setItem('state_data', JSON.stringify(this.stateData));
      }
    } catch (error) {
      console.warn('Error loading geographic data, using fallbacks:', error);
      this.countyData = this.getSampleCountyData();
      this.stateData = this.getSampleStateData();
    }
  }

  /**
   * Load historical tornado data
   */
  async loadHistoricalTornadoData() {
    try {
      // Check if we have cached data that's recent enough
      const cachedData = localStorage.getItem('historical_tornado_data');
      const cacheTimestamp = localStorage.getItem('historical_tornado_timestamp');
      
      const cacheTTL = 7 * 24 * 60 * 60 * 1000; // 1 week
      const isCacheValid = cacheTimestamp && (Date.now() - Number(cacheTimestamp)) < cacheTTL;
      
      if (cachedData && isCacheValid) {
        this.historicalData = JSON.parse(cachedData);
        this.lastUpdated.historical = Number(cacheTimestamp);
        console.log('Loaded historical tornado data from cache');
        return;
      }
      
      // Try to fetch from an API endpoint
      // For demo, we'll use static data
      this.historicalData = this.getSampleHistoricalData();
      
      // Cache the data
      localStorage.setItem('historical_tornado_data', JSON.stringify(this.historicalData));
      localStorage.setItem('historical_tornado_timestamp', Date.now().toString());
      this.lastUpdated.historical = Date.now();
    } catch (error) {
      console.warn('Error loading historical data, using fallback:', error);
      this.historicalData = this.getSampleHistoricalData();
      this.lastUpdated.historical = Date.now();
    }
  }

  /**
   * Load recent storm reports
   */
  async loadRecentStormReports() {
    try {
      // Check cache first
      const cachedReports = localStorage.getItem('recent_storm_reports');
      const cacheTimestamp = localStorage.getItem('storm_reports_timestamp');
      
      const cacheTTL = 3 * 60 * 60 * 1000; // 3 hours
      const isCacheValid = cacheTimestamp && (Date.now() - Number(cacheTimestamp)) < cacheTTL;
      
      if (cachedReports && isCacheValid) {
        this.stormReports = JSON.parse(cachedReports);
        this.lastUpdated.reports = Number(cacheTimestamp);
        console.log('Loaded storm reports from cache');
        return;
      }
      
      // Try to fetch from an API (for demo, use sample data)
      this.stormReports = this.getSampleStormReports();
      
      // Cache the data
      localStorage.setItem('recent_storm_reports', JSON.stringify(this.stormReports));
      localStorage.setItem('storm_reports_timestamp', Date.now().toString());
      this.lastUpdated.reports = Date.now();
    } catch (error) {
      console.warn('Error loading storm reports, using fallback:', error);
      this.stormReports = this.getSampleStormReports();
      this.lastUpdated.reports = Date.now();
    }
  }

  /**
   * Set up monitoring for active weather alerts
   */
  setupAlertMonitoring() {
    // Initial load
    this.fetchActiveAlerts();
    
    // Set up polling (every 15 minutes)
    setInterval(() => {
      this.fetchActiveAlerts();
    }, 15 * 60 * 1000);
  }

  /**
   * Fetch active weather alerts
   */
  async fetchActiveAlerts() {
    try {
      console.log("Fetching real alerts from NWS API");
      
      // Use the real NWS API instead of sample data
      const response = await fetch('https://api.weather.gov/alerts/active?status=actual&message_type=alert,update', {
        headers: {
          'User-Agent': 'CALM-Tornado-Prediction-App/1.0 (https://github.com/username/calm; contact@example.com)',
          'Accept': 'application/geo+json'
        }
      });
      
      if (!response.ok) {
        throw new Error(`NWS API request failed with status ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Received alert data from NWS:", data);
      
      // Process the real NWS alert data
      this.activeAlerts = this.processNWSAlerts(data);
      this.lastUpdated.alerts = Date.now();
      
      // Cache the data
      localStorage.setItem('active_weather_alerts', JSON.stringify(this.activeAlerts));
      localStorage.setItem('alerts_timestamp', Date.now().toString());
      
      return this.activeAlerts;
    } catch (error) {
      console.warn('Error fetching active alerts, using cached data:', error);
      
      // Try to use cached data
      const cachedAlerts = localStorage.getItem('active_weather_alerts');
      if (cachedAlerts) {
        this.activeAlerts = JSON.parse(cachedAlerts);
      } else {
        // If no cached data, return an empty array instead of using sample data
        this.activeAlerts = [];
      }
      
      this.lastUpdated.alerts = Date.now();
      return this.activeAlerts;
    }
  }

  /**
   * Process NWS alert data
   * @param {Object} data - Raw API response from NWS
   * @returns {Array} Processed alert objects
   */
  processNWSAlerts(data) {
    if (!data || !data.features || !Array.isArray(data.features)) {
      return [];
    }
    
    return data.features
      .filter(feature => {
        // Only include tornado and severe thunderstorm alerts
        const event = feature.properties?.event?.toLowerCase() || '';
        return event.includes('tornado') || event.includes('thunderstorm');
      })
      .map(feature => {
        const props = feature.properties || {};
        
        // Extract counties from affected zones
        const affectedZones = Array.isArray(props.affectedZones) ? props.affectedZones : [];
        const counties = [];
        const states = new Set();
        
        // Parse area description to extract counties and states
        if (props.areaDesc) {
          const areaDesc = props.areaDesc;
          
          // Extract state codes (2 uppercase letters)
          const stateMatches = areaDesc.match(/\b([A-Z]{2})\b/g) || [];
          stateMatches.forEach(state => states.add(state));
          
          // Extract county names (usually followed by "County" or "Parish")
          const countyMatches = areaDesc.match(/([A-Za-z]+)\s+(County|Parish)/g) || [];
          countyMatches.forEach(county => {
            counties.push(county.replace(/(County|Parish)/, '').trim());
          });
        }
        
        return {
          id: props.id || `alert-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
          type: props.event || 'Unknown Alert',
          issued: props.sent || new Date().toISOString(),
          expires: props.expires || new Date(Date.now() + 3600000).toISOString(),
          counties: counties,
          states: Array.from(states),
          description: props.description || '',
          urgency: props.urgency || 'Unknown',
          severity: props.severity || 'Unknown',
          certainty: props.certainty || 'Unknown',
          geometry: feature.geometry || null
        };
      });
  }

  /**
   * Get weather conditions for a specific location
   * @param {Object} location - Location with lat/lon
   * @returns {Promise<Object>} Weather conditions
   */
  async getWeatherConditions(location) {
    try {
      const cacheKey = `weather_${location.lat.toFixed(2)}_${location.lon.toFixed(2)}`;
      const cachedWeather = localStorage.getItem(cacheKey);
      const cacheTimestamp = localStorage.getItem(`${cacheKey}_timestamp`);
      
      const cacheTTL = 30 * 60 * 1000; // 30 minutes
      const isCacheValid = cacheTimestamp && (Date.now() - Number(cacheTimestamp)) < cacheTTL;
      
      if (cachedWeather && isCacheValid) {
        return JSON.parse(cachedWeather);
      }
      
      // For demo, generate realistic weather conditions based on location and date
      const weather = this.generateWeatherConditions(location);
      
      // Cache the data
      localStorage.setItem(cacheKey, JSON.stringify(weather));
      localStorage.setItem(`${cacheKey}_timestamp`, Date.now().toString());
      
      this.currentWeatherConditions[`${location.lat.toFixed(2)},${location.lon.toFixed(2)}`] = weather;
      this.lastUpdated.weather = Date.now();
      
      return weather;
    } catch (error) {
      console.warn('Error getting weather conditions, using fallback:', error);
      return this.generateWeatherConditions(location);
    }
  }

  /**
   * Get active weather alerts for a location
   * @param {Object} location - Location with lat/lon
   * @returns {Array} Active alerts for the location
   */
  getActiveAlertsForLocation(location) {
    if (!this.initialized) {
      console.warn('Data collector not initialized yet');
      return [];
    }
    
    // Find the county for this location
    const county = this.findCountyForLocation(location);
    if (!county) return [];
    
    // Filter alerts by county and state
    return this.activeAlerts.filter(alert => {
      return (
        alert.counties.includes(county.name) &&
        alert.states.includes(county.state)
      );
    });
  }

  /**
   * Find the county for a location
   * @param {Object} location - Location with lat/lon
   * @returns {Object|null} County data if found
   */
  findCountyForLocation(location) {
    if (!this.countyData || !this.countyData.features) {
      return null;
    }
    
    // This is a simplified approach; in a real implementation,
    // you would use proper geospatial containment tests
    let closestCounty = null;
    let closestDistance = Number.MAX_VALUE;
    
    for (const feature of this.countyData.features) {
      if (feature.geometry && feature.geometry.coordinates) {
        // Use the county centroid for simplicity
        const center = feature.properties.centroid || 
          [feature.geometry.coordinates[0], feature.geometry.coordinates[1]];
        
        const distance = this.calculateDistance(
          location.lat, location.lon,
          center[1], center[0]
        );
        
        if (distance < closestDistance) {
          closestDistance = distance;
          closestCounty = {
            name: feature.properties.name,
            state: feature.properties.state,
            fips: feature.properties.fips,
            lat: center[1],
            lon: center[0]
          };
        }
      }
    }
    
    // Only return if within reasonable distance (50km)
    return closestDistance < 50 ? closestCounty : null;
  }

  /**
   * Get county information for a ZIP code
   * @param {String} zipCode - ZIP code
   * @returns {Promise<Object|null>} County information
   */
  async getCountyForZipCode(zipCode) {
    try {
      // Check the ZIP code cache
      const cacheKey = `zip_${zipCode}`;
      const cachedData = localStorage.getItem(cacheKey);
      
      if (cachedData) {
        return JSON.parse(cachedData);
      }
      
      // In a real implementation, you would query an API
      // For demo, use a small set of hardcoded ZIP codes
      const zipMapping = {
        '90210': { lat: 34.1030, lon: -118.4105, county: 'Los Angeles', state: 'CA' },
        '10001': { lat: 40.7501, lon: -73.9997, county: 'New York', state: 'NY' },
        '73301': { lat: 30.2672, lon: -97.7431, county: 'Travis', state: 'TX' },
        '60601': { lat: 41.8842, lon: -87.6209, county: 'Cook', state: 'IL' },
        '33139': { lat: 25.7906, lon: -80.1300, county: 'Miami-Dade', state: 'FL' },
        '75201': { lat: 32.7864, lon: -96.7970, county: 'Dallas', state: 'TX' },
        '66101': { lat: 39.1155, lon: -94.6268, county: 'Wyandotte', state: 'KS' },
        '73102': { lat: 35.4691, lon: -97.5195, county: 'Oklahoma', state: 'OK' },
        '70112': { lat: 29.9614, lon: -90.0728, county: 'Orleans', state: 'LA' },
        '88201': { lat: 33.3943, lon: -104.5230, county: 'Chaves', state: 'NM' }
      };
      
      if (zipMapping[zipCode]) {
        const result = {
          lat: zipMapping[zipCode].lat,
          lon: zipMapping[zipCode].lon,
          name: zipMapping[zipCode].county,
          state: zipMapping[zipCode].state
        };
        
        // Cache the result
        localStorage.setItem(cacheKey, JSON.stringify(result));
        
        return result;
      }
      
      return null;
    } catch (error) {
      console.warn('Error getting county for ZIP code:', error);
      return null;
    }
  }

  /**
   * Get tornado history for a location
   * @param {Object} location - Location with lat/lon
   * @param {Number} radius - Search radius in km
   * @returns {Array} Historical tornado events
   */
  getTornadoHistoryForLocation(location, radius = 50) {
    if (!this.initialized || !this.historicalData.length) {
      return [];
    }
    
    return this.historicalData.filter(event => {
      const distance = this.calculateDistance(
        location.lat, location.lon,
        event.lat, event.lon
      );
      return distance <= radius;
    });
  }

  /**
   * Get tornado history for a county
   * @param {String} countyName - County name
   * @param {String} stateName - State name
   * @returns {Array} Historical tornado events
   */
  getTornadoHistoryForCounty(countyName, stateName) {
    if (!this.initialized || !this.historicalData.length) {
      return [];
    }
    
    return this.historicalData.filter(event => {
      return (
        event.county?.toLowerCase() === countyName.toLowerCase() &&
        event.state?.toLowerCase() === stateName.toLowerCase()
      );
    });
  }

  /**
   * Get recent storm reports near a location
   * @param {Object} location - Location with lat/lon
   * @param {Number} radius - Search radius in km
   * @returns {Array} Recent storm reports
   */
  getRecentStormReportsNear(location, radius = 50) {
    if (!this.initialized || !this.stormReports.length) {
      return [];
    }
    
    return this.stormReports.filter(report => {
      const distance = this.calculateDistance(
        location.lat, location.lon,
        report.lat, report.lon
      );
      return distance <= radius;
    });
  }

  /**
   * Verify if a tornado occurred at a specific location
   * @param {Object} location - Location with lat/lon
   * @param {Object} timeframe - Timeframe for verification
   * @returns {Promise<Boolean>} True if tornado occurred
   */
  async verifyTornadoOccurrence(location, timeframe) {
    try {
      // Check NWS storm reports or similar source
      // For demo, use a simulated approach
      const radius = 25; // km
      const reports = await this.fetchStormReportsForTimeframe(
        location, timeframe.start, timeframe.end, radius
      );
      
      // Check if any reports are tornado-related
      return reports.some(report => 
        report.event_type === 'tornado' || 
        report.event_type === 'funnel cloud'
      );
    } catch (error) {
      console.error('Error verifying tornado occurrence:', error);
      return false;
    }
  }

  /**
   * Fetch storm reports for a specific timeframe
   * @param {Object} location - Location with lat/lon
   * @param {Date} startTime - Start of timeframe
   * @param {Date} endTime - End of timeframe
   * @param {Number} radius - Search radius in km
   * @returns {Promise<Array>} Storm reports
   */
  async fetchStormReportsForTimeframe(location, startTime, endTime, radius) {
    try {
      // In a real implementation, query the NWS API or similar
      // For demo, generate some realistic reports
      const reports = this.getSampleStormReports().filter(report => {
        const reportTime = new Date(report.time);
        return (
          reportTime >= startTime &&
          reportTime <= endTime
        );
      });
      
      // Filter by radius
      return reports.filter(report => {
        const distance = this.calculateDistance(
          location.lat, location.lon,
          report.lat, report.lon
        );
        return distance <= radius;
      });
    } catch (error) {
      console.warn('Error fetching storm reports for timeframe:', error);
      return [];
    }
  }

  /**
   * Calculate distance between two coordinates
   * @param {Number} lat1 - Latitude of point 1
   * @param {Number} lon1 - Longitude of point 1
   * @param {Number} lat2 - Latitude of point 2
   * @param {Number} lon2 - Longitude of point 2
   * @returns {Number} Distance in kilometers
   */
  calculateDistance(lat1, lon1, lat2, lon2) {
    // Simplified haversine formula
    const p = 0.017453292519943295; // Math.PI / 180
    const c = Math.cos;
    const a = 0.5 - c((lat2 - lat1) * p)/2 + 
              c(lat1 * p) * c(lat2 * p) * 
              (1 - c((lon2 - lon1) * p))/2;
    
    return 12742 * Math.asin(Math.sqrt(a)); // 2 * R; R = 6371 km
  }

  /**
   * Generate weather conditions based on location and date
   * @param {Object} location - Location with lat/lon
   * @returns {Object} Weather conditions
   */
  generateWeatherConditions(location) {
    const now = new Date();
    const month = now.getMonth(); // 0-11
    const hour = now.getHours(); // 0-23
    
    // Artificial values to make them vary by location
    const locationSeed = (location.lat + location.lon) % 1;
    
    // Base temperature varies by month (Northern Hemisphere)
    let baseTemp = 15; // °C
    if (month >= 5 && month <= 8) {
      // Summer
      baseTemp = 25 + (locationSeed * 5);
    } else if (month >= 9 && month <= 11) {
      // Fall
      baseTemp = 15 + (locationSeed * 5);
    } else if (month >= 0 && month <= 2) {
      // Winter
      baseTemp = 5 + (locationSeed * 5);
    } else {
      // Spring
      baseTemp = 15 + (locationSeed * 5);
    }
    
    // Temperature varies by hour
    const hourFactor = Math.sin((hour - 14) * Math.PI / 12) * 5;
    const temperature = baseTemp + hourFactor;
    
    // Humidity based on location, season and time
    let humidity = 60 + (locationSeed * 20);
    if (month >= 5 && month <= 8) {
      // Higher humidity in summer
      humidity += 10;
    }
    
    // Pressure varies slightly (normal is around 1013.25 hPa)
    const pressure = 1013.25 + ((locationSeed - 0.5) * 10);
    
    // Wind speed and direction
    const windSpeed = 5 + (locationSeed * 15); // 5-20 km/h
    const windDirection = Math.floor(locationSeed * 360); // 0-359 degrees
    
    // Chance of precipitation
    const precipChance = locationSeed * 100;
    
    // Realistic weather condition based on inputs
    let condition = 'clear';
    if (precipChance > 80) {
      condition = 'thunderstorm';
    } else if (precipChance > 60) {
      condition = 'rain';
    } else if (precipChance > 40) {
      condition = 'cloudy';
    } else if (precipChance > 20) {
      condition = 'partly cloudy';
    }
    
    return {
      temperature: parseFloat(temperature.toFixed(1)),
      humidity: parseFloat(humidity.toFixed(1)),
      pressure: parseFloat(pressure.toFixed(1)),
      windSpeed: parseFloat(windSpeed.toFixed(1)),
      windDirection,
      condition,
      precipitationChance: parseFloat(precipChance.toFixed(1)),
      dewPoint: parseFloat((temperature - ((100 - humidity) / 5)).toFixed(1)),
      cloudCover: Math.min(100, parseFloat((precipChance * 1.2).toFixed(1))),
      timestamp: now.toISOString()
    };
  }

  /**
   * Get sample county data (simplified)
   * @returns {Object} GeoJSON-like structure with county data
   */
  getSampleCountyData() {
    return {
      type: 'FeatureCollection',
      features: [
        {
          type: 'Feature',
          properties: {
            name: 'Oklahoma',
            state: 'OK',
            fips: '40109',
            centroid: [-97.5195, 35.4691]
          },
          geometry: {
            type: 'Point',
            coordinates: [-97.5195, 35.4691]
          }
        },
        {
          type: 'Feature',
          properties: {
            name: 'Dallas',
            state: 'TX',
            fips: '48113',
            centroid: [-96.7970, 32.7864]
          },
          geometry: {
            type: 'Point',
            coordinates: [-96.7970, 32.7864]
          }
        },
        {
          type: 'Feature',
          properties: {
            name: 'Cook',
            state: 'IL',
            fips: '17031',
            centroid: [-87.6209, 41.8842]
          },
          geometry: {
            type: 'Point',
            coordinates: [-87.6209, 41.8842]
          }
        },
        {
          type: 'Feature',
          properties: {
            name: 'Orleans',
            state: 'LA',
            fips: '22071',
            centroid: [-90.0728, 29.9614]
          },
          geometry: {
            type: 'Point',
            coordinates: [-90.0728, 29.9614]
          }
        },
        {
          type: 'Feature',
          properties: {
            name: 'Chaves',
            state: 'NM',
            fips: '35005',
            centroid: [-104.5230, 33.3943]
          },
          geometry: {
            type: 'Point',
            coordinates: [-104.5230, 33.3943]
          }
        }
        // In a real implementation, this would contain all US counties
      ]
    };
  }

  /**
   * Get sample state data (simplified)
   * @returns {Object} GeoJSON-like structure with state data
   */
  getSampleStateData() {
    return {
      type: 'FeatureCollection',
      features: [
        {
          type: 'Feature',
          properties: {
            name: 'Oklahoma',
            abbr: 'OK',
            fips: '40',
            centroid: [-97.5033, 35.5837]
          },
          geometry: {
            type: 'Point',
            coordinates: [-97.5033, 35.5837]
          }
        },
        {
          type: 'Feature',
          properties: {
            name: 'Texas',
            abbr: 'TX',
            fips: '48',
            centroid: [-99.3312, 31.4757]
          },
          geometry: {
            type: 'Point',
            coordinates: [-99.3312, 31.4757]
          }
        },
        {
          type: 'Feature',
          properties: {
            name: 'Kansas',
            abbr: 'KS',
            fips: '20',
            centroid: [-98.3822, 38.4937]
          },
          geometry: {
            type: 'Point',
            coordinates: [-98.3822, 38.4937]
          }
        },
        {
          type: 'Feature',
          properties: {
            name: 'Illinois',
            abbr: 'IL',
            fips: '17',
            centroid: [-89.1991, 40.0632]
          },
          geometry: {
            type: 'Point',
            coordinates: [-89.1991, 40.0632]
          }
        },
        {
          type: 'Feature',
          properties: {
            name: 'Louisiana',
            abbr: 'LA',
            fips: '22',
            centroid: [-91.9623, 31.0689]
          },
          geometry: {
            type: 'Point',
            coordinates: [-91.9623, 31.0689]
          }
        }
        // In a real implementation, this would contain all US states
      ]
    };
  }

  /**
   * Get sample historical tornado data
   * @returns {Array} Sample tornado events
   */
  getSampleHistoricalData() {
    // Return a realistic set of historical tornado events
    return [
      {
        id: 'tor-2020-001',
        date: '2020-05-10T15:30:00Z',
        lat: 35.4691,
        lon: -97.5195,
        county: 'Oklahoma',
        state: 'OK',
        magnitude: 3, // EF3
        path_length: 12.5, // km
        path_width: 400, // meters
        fatalities: 0,
        injuries: 7,
        damage_estimate: 2500000, // USD
        time_on_ground: 25 // minutes
      },
      {
        id: 'tor-2019-023',
        date: '2019-04-23T14:15:00Z',
        lat: 33.3943,
        lon: -104.5230,
        county: 'Chaves',
        state: 'NM',
        magnitude: 2, // EF2
        path_length: 8.2, // km
        path_width: 250, // meters
        fatalities: 0,
        injuries: 3,
        damage_estimate: 1200000, // USD
        time_on_ground: 18 // minutes
      },
      {
        id: 'tor-2021-017',
        date: '2021-03-28T18:40:00Z',
        lat: 32.7864,
        lon: -96.7970,
        county: 'Dallas',
        state: 'TX',
        magnitude: 1, // EF1
        path_length: 5.4, // km
        path_width: 150, // meters
        fatalities: 0,
        injuries: 1,
        damage_estimate: 500000, // USD
        time_on_ground: 12 // minutes
      },
      {
        id: 'tor-2022-009',
        date: '2022-04-15T19:20:00Z',
        lat: 38.4937,
        lon: -98.3822,
        county: 'Barton',
        state: 'KS',
        magnitude: 4, // EF4
        path_length: 28.7, // km
        path_width: 550, // meters
        fatalities: 3,
        injuries: 21,
        damage_estimate: 15000000, // USD
        time_on_ground: 45 // minutes
      },
      {
        id: 'tor-2023-032',
        date: '2023-05-05T16:10:00Z',
        lat: 29.9614,
        lon: -90.0728,
        county: 'Orleans',
        state: 'LA',
        magnitude: 0, // EF0
        path_length: 2.1, // km
        path_width: 80, // meters
        fatalities: 0,
        injuries: 0,
        damage_estimate: 50000, // USD
        time_on_ground: 6 // minutes
      }
      // In a real implementation, this would contain hundreds or thousands of records
    ];
  }

  /**
   * Get sample storm reports
   * @returns {Array} Sample storm reports
   */
  getSampleStormReports() {
    const now = new Date();
    const oneHourAgo = new Date(now.getTime() - 60 * 60 * 1000);
    const twoHoursAgo = new Date(now.getTime() - 2 * 60 * 60 * 1000);
    const threeHoursAgo = new Date(now.getTime() - 3 * 60 * 60 * 1000);
    
    return [
      {
        id: 'rep-001',
        time: threeHoursAgo.toISOString(),
        lat: 35.4691,
        lon: -97.5195,
        county: 'Oklahoma',
        state: 'OK',
        event_type: 'hail',
        description: '1.5 inch hail reported',
        source: 'trained spotter'
      },
      {
        id: 'rep-002',
        time: twoHoursAgo.toISOString(),
        lat: 35.4700,
        lon: -97.5150,
        county: 'Oklahoma',
        state: 'OK',
        event_type: 'tornado',
        description: 'Brief tornado touchdown reported near I-35',
        source: 'law enforcement'
      },
      {
        id: 'rep-003',
        time: oneHourAgo.toISOString(),
        lat: 32.7864,
        lon: -96.7970,
        county: 'Dallas',
        state: 'TX',
        event_type: 'wind damage',
        description: 'Trees down, measured 65mph gust',
        source: 'mesonet'
      },
      {
        id: 'rep-004',
        time: now.toISOString(),
        lat: 33.3943,
        lon: -104.5230,
        county: 'Chaves',
        state: 'NM',
        event_type: 'funnel cloud',
        description: 'Funnel cloud observed but no touchdown',
        source: 'public'
      }
    ];
  }
}

// Export the class
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TornadoDataCollector;
} 
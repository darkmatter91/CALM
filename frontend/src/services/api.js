import axios from 'axios';

// Determine the best API base URL to use
const getApiBaseUrl = () => {
  // Production mode
  if (process.env.NODE_ENV === 'production') {
    return '';
  }
  
  // Development mode
  const hostname = window.location.hostname;
  
  // If accessed from another device (not localhost), use the hostname
  if (hostname !== 'localhost' && hostname !== '127.0.0.1') {
    return `http://${hostname}:5000`;
  }
  
  // Default for local development
  return 'http://localhost:5000';
};

// Create an instance of axios with a base URL
const API = axios.create({
  baseURL: getApiBaseUrl(),
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // Increase timeout to 30 seconds for long-running processes
  withCredentials: false // Set to true if your API requires cookies
});

// Add request interceptor for debugging
API.interceptors.request.use(
  config => {
    console.log(`[API Request] ${config.method.toUpperCase()} ${config.url}`, config.params || config.data);
    return config;
  },
  error => {
    console.error('[API Request Error]', error);
    return Promise.reject(error);
  }
);

// Add response interceptor for debugging
API.interceptors.response.use(
  response => {
    console.log(`[API Response] ${response.status} from ${response.config.url}`, response.data);
    return response;
  },
  error => {
    if (error.response) {
      // Server responded with an error status code
      console.error(`[API Error] ${error.response.status} from ${error.config.url}:`, 
                   error.response.data);
    } else if (error.request) {
      // Request made but no response received
      console.error('[API Error] No response received:', error.request);
    } else {
      // Error in setting up the request
      console.error('[API Error] Request setup error:', error.message);
    }
    return Promise.reject(error);
  }
);

// API functions
const weatherService = {
  // Health check
  healthCheck: async () => {
    try {
      const response = await API.get('/api/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  },

  // Get predictions based on location (ZIP, coordinates, etc.)
  getPredictions: async (params) => {
    try {
      const response = await API.get('/api/predict', { params });
      return response.data;
    } catch (error) {
      console.error('Failed to get predictions:', error);
      throw error;
    }
  },

  // Get tornado predictions
  getTornadoPredictions: async (options = {}) => {
    const { useFastMode = true } = options;
    try {
      const response = await API.get('/api/tornado/predictions', { 
        params: { 
          fast: useFastMode
        } 
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get tornado predictions:', error);
      throw error;
    }
  },

  // Get prediction stats
  getPredictionStats: async () => {
    try {
      const response = await API.get('/api/tornado/stats');
      return response.data;
    } catch (error) {
      console.error('Failed to get prediction stats:', error);
      throw error;
    }
  },

  // Get weather alerts
  getWeatherAlerts: async (params) => {
    try {
      const response = await API.get('/api/weather/alerts', { params });
      return response.data;
    } catch (error) {
      console.error('Failed to get weather alerts:', error);
      throw error;
    }
  },

  // Get live radar data
  getLiveRadar: async (params) => {
    try {
      const response = await API.get('/api/radar/live', { params });
      return response.data;
    } catch (error) {
      console.error('Failed to get live radar:', error);
      throw error;
    }
  },

  // Post to process radar images
  processRadar: async (data) => {
    try {
      const response = await API.post('/api/radar', data);
      return response.data;
    } catch (error) {
      console.error('Failed to process radar:', error);
      throw error;
    }
  },

  // Post to analyze radar patterns
  analyzeRadarPatterns: async (data) => {
    try {
      const response = await API.post('/api/radar/analyze', data);
      return response.data;
    } catch (error) {
      console.error('Failed to analyze radar patterns:', error);
      throw error;
    }
  },

  // Post to predict with AI
  predictWithAI: async (options = {}) => {
    try {
      const { 
        radar_data, 
        fast_mode = true, 
        zipcode = null, 
        latitude = null, 
        longitude = null,
        coverage = null, 
        country = null
      } = options;
      
      // Prepare the request data
      const requestData = { radar_data, fast_mode };
      
      // Add location data if provided
      if (zipcode) requestData.zipcode = zipcode;
      if (latitude && longitude) {
        requestData.latitude = latitude;
        requestData.longitude = longitude;
      }
      
      // Add national coverage options if specified
      if (coverage === 'national') {
        requestData.national_coverage = true;
        if (country) requestData.country = country;
      }
      
      const response = await API.post('/api/predict/ai', requestData);
      return response.data;
    } catch (error) {
      console.error('Failed to get AI prediction:', error);
      throw error;
    }
  },

  // Get model status
  getModelStatus: async () => {
    try {
      const response = await API.get('/api/model/status');
      return response.data;
    } catch (error) {
      console.error('Failed to get model status:', error);
      throw error;
    }
  },

  // Post to train model
  trainModel: async (data) => {
    try {
      const response = await API.post('/api/model/train', data);
      return response.data;
    } catch (error) {
      console.error('Failed to train model:', error);
      throw error;
    }
  }
};

export default weatherService; 
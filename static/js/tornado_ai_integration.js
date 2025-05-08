/**
 * Tornado AI Integration
 * Connects the tornado data collector with the AI prediction model and UI
 */

class TornadoAIIntegration {
    constructor() {
        // References to other components
        this.dataCollector = null;
        this.predictionModel = null;
        
        // Configuration
        this.config = {
            modelUpdateInterval: 7 * 24 * 60 * 60 * 1000, // Update model weekly
            predictionCacheTime: 6 * 60 * 60 * 1000,      // Cache predictions for 6 hours
            minimumTrainingExamples: 50                   // Minimum examples before making predictions
        };
        
        // Cache for predictions
        this.predictionCache = {};
        
        // Tracking for model updates
        this.lastModelUpdate = null;
    }
    
    /**
     * Initialize the integration system
     */
    async initialize() {
        try {
            console.log("Initializing Tornado AI Integration...");
            
            // Initialize data collector
            if (window.tornadoDataCollector) {
                this.dataCollector = window.tornadoDataCollector;
                await this.dataCollector.initialize();
                console.log("Data collector initialized");
            } else {
                console.error("Tornado data collector not found");
                return false;
            }
            
            // Initialize prediction model
            if (window.tornadoPredictionModel) {
                this.predictionModel = window.tornadoPredictionModel;
                await this.predictionModel.initialize();
                console.log("Prediction model initialized");
            } else {
                console.error("Tornado prediction model not found");
                return false;
            }
            
            // Get last model update time
            this.lastModelUpdate = localStorage.getItem('lastTornadoModelUpdate') ? 
                new Date(localStorage.getItem('lastTornadoModelUpdate')) : null;
            
            // Check if model needs update
            await this.checkModelUpdate();
            
            // Setup scheduled tasks
            this.setupScheduledTasks();
            
            console.log("Tornado AI Integration initialized successfully");
            return true;
        } catch (error) {
            console.error("Error initializing Tornado AI Integration:", error);
            return false;
        }
    }
    
    /**
     * Setup scheduled background tasks
     */
    setupScheduledTasks() {
        // Check for verified predictions daily
        setInterval(() => this.processVerifiedPredictions(), 24 * 60 * 60 * 1000);
        
        // Update model weekly if we have new training data
        setInterval(() => this.checkModelUpdate(), this.config.modelUpdateInterval);
        
        // Clear old prediction cache entries hourly
        setInterval(() => this.cleanPredictionCache(), 60 * 60 * 1000);
    }
    
    /**
     * Make a tornado prediction for a given location and weather conditions
     */
    async predictTornado(location, weatherData) {
        try {
            if (!this.dataCollector || !this.predictionModel) {
                console.error("AI system not fully initialized");
                return null;
            }
            
            // Check if we have enough training data to make predictions
            const metrics = this.predictionModel.getMetrics();
            if (metrics.trainingExamples < this.config.minimumTrainingExamples) {
                console.log("Not enough training data for predictions yet");
                return this.getFallbackPrediction(location, weatherData);
            }
            
            // Create a cache key based on location and current time (rounded to nearest hour)
            const currentHour = new Date();
            currentHour.setMinutes(0, 0, 0);
            const cacheKey = `${location.lat.toFixed(2)}_${location.lon.toFixed(2)}_${currentHour.getTime()}`;
            
            // Check cache first
            if (this.predictionCache[cacheKey]) {
                console.log("Using cached tornado prediction");
                return this.predictionCache[cacheKey];
            }
            
            // Prepare features for the model
            const features = this.prepareFeatures(location, weatherData);
            
            // Make prediction
            const prediction = await this.predictionModel.predict(features);
            
            if (!prediction) {
                console.error("Failed to get prediction from model");
                return this.getFallbackPrediction(location, weatherData);
            }
            
            // Enhance the prediction with additional information
            const enhancedPrediction = {
                ...prediction,
                
                // Add location information
                location: {
                    lat: location.lat,
                    lon: location.lon,
                    zipcode: location.zipcode,
                    city: location.city,
                    state: location.state
                },
                
                // Add timestamp and expiration
                timestamp: new Date().toISOString(),
                expiresAt: new Date(Date.now() + 6 * 60 * 60 * 1000).toISOString(),
                
                // Add storm motion data if available
                stormMotion: weatherData.wind_direction || null,
                stormSpeed: weatherData.wind_speed || null,
                
                // Add status tracking
                status: "active",
                verified: false,
                
                // Store original input features for verification/learning
                features: features
            };
            
            // Cache the prediction
            this.predictionCache[cacheKey] = enhancedPrediction;
            
            // Store prediction for verification/training
            this.storeForVerification(enhancedPrediction);
            
            return enhancedPrediction;
        } catch (error) {
            console.error("Error predicting tornado:", error);
            return this.getFallbackPrediction(location, weatherData);
        }
    }
    
    /**
     * Generate a fallback prediction when the model can't provide one
     */
    getFallbackPrediction(location, weatherData) {
        // Simple heuristic-based fallback
        let probability = 0.01;  // Base probability is low
        
        // Increase based on weather factors known to contribute to tornado formation
        if (weatherData.cape > 1000) probability += 0.05;
        if (weatherData.cape > 2000) probability += 0.1;
        
        if (weatherData.wind_shear > 30) probability += 0.05;
        if (weatherData.wind_shear > 50) probability += 0.1;
        
        if (weatherData.helicity > 150) probability += 0.05;
        if (weatherData.helicity > 300) probability += 0.1;
        
        // Check if we're in tornado alley during tornado season
        const month = weatherData.month || (new Date().getMonth() + 1);
        const isTornadoSeason = month >= 3 && month <= 6;  // March-June
        
        // Check if location is in tornado-prone area
        const isInTornadoAlley = this.isLocationInTornadoAlley(location);
        
        if (isTornadoSeason && isInTornadoAlley) {
            probability += 0.1;
        }
        
        // Cap at reasonable value
        probability = Math.min(probability, 0.5);
        
        // Format similar to model prediction
        return {
            probability: probability,
            confidence: 0.3,  // Low confidence for heuristic prediction
            riskLevel: this.getRiskLevelFromProbability(probability),
            tornado: probability >= 0.4,
            
            // Add location information
            location: {
                lat: location.lat,
                lon: location.lon,
                zipcode: location.zipcode,
                city: location.city,
                state: location.state
            },
            
            // Add timestamp and expiration
            timestamp: new Date().toISOString(),
            expiresAt: new Date(Date.now() + 6 * 60 * 60 * 1000).toISOString(),
            
            // Add storm motion data if available
            stormMotion: weatherData.wind_direction || null,
            stormSpeed: weatherData.wind_speed || null,
            
            // Mark as fallback
            isFallback: true,
            
            // Add status tracking
            status: "active",
            verified: false
        };
    }
    
    /**
     * Convert risk level to probability
     */
    getRiskLevelFromProbability(probability) {
        if (probability < 0.2) {
            return 'Low';
        } else if (probability < 0.4) {
            return 'Moderate';
        } else if (probability < 0.6) {
            return 'High';
        } else if (probability < 0.8) {
            return 'Very High';
        } else {
            return 'Extreme';
        }
    }
    
    /**
     * Check if location is in the US tornado alley or other tornado-prone regions
     */
    isLocationInTornadoAlley(location) {
        // Rough bounding boxes for tornado-prone regions
        const tornadoRegions = [
            // Traditional Tornado Alley (Central US)
            {
                minLat: 30, maxLat: 45,
                minLon: -100, maxLon: -85
            },
            // Dixie Alley (Southeast US)
            {
                minLat: 31, maxLat: 36.5,
                minLon: -92, maxLon: -82
            },
            // Hoosier Alley (Midwest)
            {
                minLat: 37, maxLat: 43,
                minLon: -90, maxLon: -80
            }
        ];
        
        // Check if location is in any defined tornado-prone region
        return tornadoRegions.some(region => 
            location.lat >= region.minLat && location.lat <= region.maxLat &&
            location.lon >= region.minLon && location.lon <= region.maxLon
        );
    }
    
    /**
     * Prepare feature vector for AI model from weather data and location
     */
    prepareFeatures(location, weatherData) {
        const now = new Date();
        
        return {
            cape: weatherData.cape || 0,
            wind_shear: weatherData.wind_shear || 0,
            helicity: weatherData.helicity || 0,
            dewpoint: weatherData.dewpoint || 0,
            temperature: weatherData.temperature || 0,
            humidity: weatherData.humidity || 0,
            month: weatherData.month || (now.getMonth() + 1),
            hour: weatherData.hour || now.getHours(),
            lat: location.lat || 0,
            lon: location.lon || 0,
            pressure_change: weatherData.pressure_change || 0,
            wind_direction: weatherData.wind_direction || 0
        };
    }
    
    /**
     * Store a prediction for later verification and model training
     */
    storeForVerification(prediction) {
        try {
            // Get existing predictions awaiting verification
            const pendingVerifications = JSON.parse(localStorage.getItem('pendingTornadoVerifications') || '[]');
            
            // Add this prediction
            pendingVerifications.push({
                ...prediction,
                verificationScheduled: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString()
            });
            
            // Limit the number stored to prevent localStorage bloat
            while (pendingVerifications.length > 100) {
                pendingVerifications.shift();
            }
            
            // Save back to storage
            localStorage.setItem('pendingTornadoVerifications', JSON.stringify(pendingVerifications));
            
        } catch (error) {
            console.error("Error storing prediction for verification:", error);
        }
    }
    
    /**
     * Process predictions that are ready for verification
     */
    async processVerifiedPredictions() {
        try {
            console.log("Processing verified predictions...");
            
            // Get pending verifications
            const pendingVerifications = JSON.parse(localStorage.getItem('pendingTornadoVerifications') || '[]');
            const now = new Date();
            
            // Separate predictions ready for verification
            const readyForVerification = [];
            const stillPending = [];
            
            pendingVerifications.forEach(prediction => {
                const verificationDate = new Date(prediction.verificationScheduled);
                if (verificationDate <= now) {
                    readyForVerification.push(prediction);
                } else {
                    stillPending.push(prediction);
                }
            });
            
            console.log(`Found ${readyForVerification.length} predictions ready for verification`);
            
            // Verify each prediction
            for (const prediction of readyForVerification) {
                await this.verifyPrediction(prediction);
            }
            
            // Save remaining pending verifications
            localStorage.setItem('pendingTornadoVerifications', JSON.stringify(stillPending));
            
        } catch (error) {
            console.error("Error processing verified predictions:", error);
        }
    }
    
    /**
     * Verify a prediction against actual tornado occurrence data
     */
    async verifyPrediction(prediction) {
        try {
            console.log(`Verifying prediction for ${prediction.location.city}, ${prediction.location.state}`);
            
            // Get the date range to check
            const predictionDate = new Date(prediction.timestamp);
            const endDate = new Date(prediction.expiresAt);
            
            // Query NWS or other sources for tornado reports in this area and time period
            // For demonstration, we'll simulate this with a data collector method
            const actualTornado = await this.checkForActualTornado(
                prediction.location, 
                predictionDate, 
                endDate
            );
            
            console.log(`Verification result: Tornado occurred = ${actualTornado}`);
            
            // Update the model with this feedback
            await this.predictionModel.updateWithFeedback(prediction, actualTornado);
            
            // Add to verified predictions log
            this.logVerifiedPrediction(prediction, actualTornado);
            
            return actualTornado;
        } catch (error) {
            console.error("Error verifying prediction:", error);
            return null;
        }
    }
    
    /**
     * Check if a tornado actually occurred based on weather service data
     */
    async checkForActualTornado(location, startDate, endDate) {
        try {
            // In a real implementation, this would query NWS API or other data sources
            // For demonstration, we'll use the data collector
            
            if (this.dataCollector) {
                const recentTornados = await this.dataCollector.getRecentTornadoEvents(7); // past week
                
                // Check if any tornado reports match this location and time
                return recentTornados.some(tornado => {
                    // Check date range
                    const tornadoDate = new Date(tornado.date);
                    const isInTimeRange = tornadoDate >= startDate && tornadoDate <= endDate;
                    
                    // Check location (within ~25 miles)
                    const distance = this.calculateDistance(
                        location.lat, location.lon,
                        tornado.lat, tornado.lon
                    );
                    
                    const isNearby = distance < 40; // km (~25 miles)
                    
                    return isInTimeRange && isNearby;
                });
            }
            
            // If no data collector or no tornado data, simulate with random outcome
            // weighted by the original prediction probability
            const randomFactor = Math.random();
            const threshold = prediction.probability * 1.5; // Adjust to match expected frequency
            
            return randomFactor < threshold;
        } catch (error) {
            console.error("Error checking for actual tornado:", error);
            return false;
        }
    }
    
    /**
     * Calculate distance between two coordinates in kilometers
     */
    calculateDistance(lat1, lon1, lat2, lon2) {
        const R = 6371; // Earth's radius in km
        const dLat = this.deg2rad(lat2 - lat1);
        const dLon = this.deg2rad(lon2 - lon1);
        
        const a = 
            Math.sin(dLat/2) * Math.sin(dLat/2) +
            Math.cos(this.deg2rad(lat1)) * Math.cos(this.deg2rad(lat2)) * 
            Math.sin(dLon/2) * Math.sin(dLon/2);
        
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
        return R * c;
    }
    
    /**
     * Convert degrees to radians
     */
    deg2rad(deg) {
        return deg * (Math.PI/180);
    }
    
    /**
     * Log verified prediction for analytics
     */
    logVerifiedPrediction(prediction, actualTornado) {
        try {
            // Get existing verified predictions
            const verifiedPredictions = JSON.parse(
                localStorage.getItem('verifiedTornadoPredictions') || '[]'
            );
            
            // Add this prediction with verification result
            verifiedPredictions.push({
                ...prediction,
                actualTornado: actualTornado,
                verifiedAt: new Date().toISOString()
            });
            
            // Keep most recent 100
            while (verifiedPredictions.length > 100) {
                verifiedPredictions.shift();
            }
            
            // Save back to storage
            localStorage.setItem('verifiedTornadoPredictions', JSON.stringify(verifiedPredictions));
            
            // Update UI metrics display if exists
            if (typeof updateModelLearningStats === 'function') {
                updateModelLearningStats();
            }
        } catch (error) {
            console.error("Error logging verified prediction:", error);
        }
    }
    
    /**
     * Clean old entries from prediction cache
     */
    cleanPredictionCache() {
        try {
            const now = Date.now();
            
            // Remove expired predictions from cache
            Object.keys(this.predictionCache).forEach(key => {
                const prediction = this.predictionCache[key];
                const expiresAt = new Date(prediction.expiresAt).getTime();
                
                if (expiresAt < now) {
                    delete this.predictionCache[key];
                }
            });
        } catch (error) {
            console.error("Error cleaning prediction cache:", error);
        }
    }
    
    /**
     * Check if model needs to be updated with new training data
     */
    async checkModelUpdate() {
        try {
            // If no last update time or it's been more than a week
            const shouldUpdate = 
                !this.lastModelUpdate || 
                (Date.now() - this.lastModelUpdate.getTime() > this.config.modelUpdateInterval);
            
            if (shouldUpdate && this.dataCollector && this.predictionModel) {
                console.log("Checking for model update with new training data...");
                
                // Get new training data
                const trainingData = this.dataCollector.getTrainingData();
                
                if (trainingData && trainingData.length > this.config.minimumTrainingExamples) {
                    console.log(`Updating model with ${trainingData.length} training examples`);
                    
                    // Update the model
                    await this.predictionModel.trainOnNewData(trainingData);
                    
                    // Update last update time
                    this.lastModelUpdate = new Date();
                    localStorage.setItem('lastTornadoModelUpdate', this.lastModelUpdate.toISOString());
                    
                    console.log("Model update complete");
                } else {
                    console.log("Not enough training data for model update");
                }
            }
        } catch (error) {
            console.error("Error checking for model update:", error);
        }
    }
    
    /**
     * Get model performance statistics for display
     */
    getModelStats() {
        try {
            // Get basic metrics from model
            const metrics = this.predictionModel ? this.predictionModel.getMetrics() : {
                accuracy: 0,
                precision: 0,
                recall: 0,
                f1Score: 0,
                trainingExamples: 0,
                lastTrainingDate: null
            };
            
            // Get verified predictions for additional stats
            const verifiedPredictions = JSON.parse(
                localStorage.getItem('verifiedTornadoPredictions') || '[]'
            );
            
            // Calculate recent performance
            let recentCorrect = 0;
            let recentTotal = 0;
            
            if (verifiedPredictions.length > 0) {
                // Use last 30 days
                const thirtyDaysAgo = new Date();
                thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
                
                const recentPredictions = verifiedPredictions.filter(p => {
                    const verifiedAt = new Date(p.verifiedAt);
                    return verifiedAt >= thirtyDaysAgo;
                });
                
                recentTotal = recentPredictions.length;
                
                recentCorrect = recentPredictions.filter(p => {
                    return (p.tornado && p.actualTornado) || (!p.tornado && !p.actualTornado);
                }).length;
            }
            
            return {
                ...metrics,
                recentAccuracy: recentTotal > 0 ? recentCorrect / recentTotal : 0,
                recentPredictions: recentTotal,
                totalVerified: verifiedPredictions.length,
                lastUpdateTime: this.lastModelUpdate ? this.lastModelUpdate.toISOString() : null
            };
        } catch (error) {
            console.error("Error getting model stats:", error);
            return {
                accuracy: 0,
                precision: 0,
                recall: 0,
                f1Score: 0,
                trainingExamples: 0,
                lastTrainingDate: null,
                recentAccuracy: 0,
                recentPredictions: 0,
                totalVerified: 0,
                lastUpdateTime: null
            };
        }
    }
}

// Create singleton instance
const tornadoAI = new TornadoAIIntegration();

// Export to window
window.tornadoAI = tornadoAI; 
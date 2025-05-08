/**
 * AI Tornado Prediction Model
 * Uses TensorFlow.js to predict tornado likelihood based on weather data
 */

class TornadoPredictionModel {
    constructor() {
        // The actual TensorFlow model
        this.model = null;
        
        // Feature names for the model input
        this.featureNames = [
            'cape', 'wind_shear', 'helicity', 'dewpoint', 
            'temperature', 'humidity', 'month', 'hour',
            'lat', 'lon', 'pressure_change', 'wind_direction'
        ];
        
        // Store feature statistics for normalization
        this.featureStats = {};
        
        // Model metrics
        this.metrics = {
            accuracy: 0,
            precision: 0,
            recall: 0,
            f1Score: 0,
            trainingExamples: 0,
            lastTrainingDate: null
        };
        
        // Database name for IndexedDB
        this.dbName = 'tornado_prediction_model';
    }
    
    /**
     * Initialize the model
     */
    async initialize() {
        try {
            console.log("Initializing tornado prediction model...");
            
            // Make sure TensorFlow.js is loaded
            if (!window.tf) {
                console.error("TensorFlow.js is not loaded. Please include it in your HTML.");
                this.loadTensorFlow();
                return false;
            }
            
            // Try to load saved model from IndexedDB
            try {
                this.model = await tf.loadLayersModel('indexeddb://' + this.dbName);
                console.log("Loaded saved model from IndexedDB");
                
                // Load feature stats from localStorage
                const savedStats = localStorage.getItem('tornadoFeatureStats');
                if (savedStats) {
                    this.featureStats = JSON.parse(savedStats);
                }
                
                // Load metrics from localStorage
                const savedMetrics = localStorage.getItem('tornadoModelMetrics');
                if (savedMetrics) {
                    this.metrics = JSON.parse(savedMetrics);
                }
                
                console.log("Model initialized successfully");
                return true;
            } catch (error) {
                console.log("No saved model found, creating a new one");
                // Create a new model
                return await this.createNewModel();
            }
        } catch (error) {
            console.error("Error initializing tornado prediction model:", error);
            return false;
        }
    }
    
    /**
     * Load TensorFlow.js dynamically if not present
     */
    loadTensorFlow() {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.18.0/dist/tf.min.js';
            script.onload = () => {
                console.log("TensorFlow.js loaded successfully");
                resolve(true);
            };
            script.onerror = () => {
                console.error("Failed to load TensorFlow.js");
                reject(false);
            };
            document.head.appendChild(script);
        });
    }
    
    /**
     * Create a new model
     */
    async createNewModel() {
        try {
            console.log("Creating new tornado prediction model");
            
            // Define the model architecture
            this.model = tf.sequential();
            
            // Input layer
            this.model.add(tf.layers.dense({
                inputShape: [this.featureNames.length],
                units: 32,
                activation: 'relu'
            }));
            
            // Hidden layers
            this.model.add(tf.layers.dropout({ rate: 0.2 }));
            this.model.add(tf.layers.dense({
                units: 16,
                activation: 'relu'
            }));
            this.model.add(tf.layers.dropout({ rate: 0.1 }));
            
            // Output layer - binary classification (tornado/no tornado)
            this.model.add(tf.layers.dense({
                units: 1,
                activation: 'sigmoid'
            }));
            
            // Compile the model
            this.model.compile({
                optimizer: tf.train.adam(0.001),
                loss: 'binaryCrossentropy',
                metrics: ['accuracy']
            });
            
            // Train the model with initial data if available
            if (window.tornadoDataCollector) {
                await this.trainInitialModel();
            }
            
            // Save the model to IndexedDB
            await this.saveModel();
            
            console.log("New model created successfully");
            return true;
        } catch (error) {
            console.error("Error creating new model:", error);
            return false;
        }
    }
    
    /**
     * Train the model with initial synthetic data
     */
    async trainInitialModel() {
        try {
            console.log("Training initial model with synthetic data");
            
            // Get training data from data collector
            const trainingData = window.tornadoDataCollector.getTrainingData();
            if (!trainingData || trainingData.length === 0) {
                console.log("No training data available");
                return false;
            }
            
            // Calculate feature stats for normalization
            this.calculateFeatureStats(trainingData);
            
            // Convert training data to tensors
            const { xs, ys } = this.convertToTensors(trainingData);
            
            // Define training parameters
            const batchSize = 32;
            const epochs = 10;
            
            // Train the model
            const history = await this.model.fit(xs, ys, {
                batchSize,
                epochs,
                validationSplit: 0.2,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        console.log(`Epoch ${epoch + 1} of ${epochs}:`);
                        console.log(`  Loss: ${logs.loss.toFixed(4)}, Accuracy: ${logs.acc.toFixed(4)}`);
                    }
                }
            });
            
            // Calculate metrics on the training data
            this.calculateMetrics(trainingData);
            
            // Update metrics
            this.metrics.trainingExamples = trainingData.length;
            this.metrics.lastTrainingDate = new Date().toISOString();
            
            // Save metrics to localStorage
            localStorage.setItem('tornadoModelMetrics', JSON.stringify(this.metrics));
            
            // Dispose tensors to free memory
            xs.dispose();
            ys.dispose();
            
            console.log("Initial model training complete");
            return true;
        } catch (error) {
            console.error("Error training initial model:", error);
            return false;
        }
    }
    
    /**
     * Save the model to IndexedDB
     */
    async saveModel() {
        try {
            if (!this.model) {
                console.error("No model to save");
                return false;
            }
            
            await this.model.save('indexeddb://' + this.dbName);
            
            // Save feature stats to localStorage
            localStorage.setItem('tornadoFeatureStats', JSON.stringify(this.featureStats));
            
            // Save metrics to localStorage
            localStorage.setItem('tornadoModelMetrics', JSON.stringify(this.metrics));
            
            console.log("Model saved successfully");
            return true;
        } catch (error) {
            console.error("Error saving model:", error);
            return false;
        }
    }
    
    /**
     * Calculate statistics for each feature for normalization
     */
    calculateFeatureStats(trainingData) {
        try {
            // Initialize feature stats
            this.featureStats = {};
            
            // For each feature, calculate min, max, mean, std
            this.featureNames.forEach(feature => {
                const values = trainingData.map(example => example.features[feature]);
                
                const min = Math.min(...values);
                const max = Math.max(...values);
                const sum = values.reduce((a, b) => a + b, 0);
                const mean = sum / values.length;
                
                // Calculate standard deviation
                const squareDiffs = values.map(value => {
                    const diff = value - mean;
                    return diff * diff;
                });
                const avgSquareDiff = squareDiffs.reduce((a, b) => a + b, 0) / values.length;
                const std = Math.sqrt(avgSquareDiff);
                
                this.featureStats[feature] = { min, max, mean, std };
            });
            
            // Save feature stats to localStorage
            localStorage.setItem('tornadoFeatureStats', JSON.stringify(this.featureStats));
            
            return true;
        } catch (error) {
            console.error("Error calculating feature stats:", error);
            return false;
        }
    }
    
    /**
     * Normalize input features using min-max normalization or z-score normalization
     */
    normalizeFeatures(features) {
        try {
            const normalizedFeatures = {};
            
            Object.keys(features).forEach(feature => {
                if (this.featureStats[feature]) {
                    // Use z-score normalization (subtract mean, divide by std)
                    const { mean, std } = this.featureStats[feature];
                    normalizedFeatures[feature] = std > 0 ? 
                        (features[feature] - mean) / std : 
                        features[feature] - mean;
                } else {
                    // If no stats, just use the raw value
                    normalizedFeatures[feature] = features[feature];
                }
            });
            
            return normalizedFeatures;
        } catch (error) {
            console.error("Error normalizing features:", error);
            return features; // Return original features on error
        }
    }
    
    /**
     * Convert training data to tensors for TensorFlow.js
     */
    convertToTensors(trainingData) {
        try {
            // Extract features and labels
            const featureVectors = trainingData.map(example => {
                // Normalize features
                const normalizedFeatures = this.normalizeFeatures(example.features);
                
                // Create feature vector in the correct order
                return this.featureNames.map(name => normalizedFeatures[name]);
            });
            
            const labels = trainingData.map(example => example.tornado ? 1 : 0);
            
            // Convert to tensors
            const xs = tf.tensor2d(featureVectors);
            const ys = tf.tensor2d(labels, [labels.length, 1]);
            
            return { xs, ys };
        } catch (error) {
            console.error("Error converting to tensors:", error);
            return null;
        }
    }
    
    /**
     * Predict tornado likelihood from weather data
     */
    async predict(weatherData) {
        try {
            if (!this.model) {
                console.error("Model not initialized");
                return null;
            }
            
            // Normalize the input features
            const normalizedFeatures = this.normalizeFeatures(weatherData);
            
            // Create feature vector in the correct order
            const featureVector = this.featureNames.map(name => 
                normalizedFeatures[name] !== undefined ? normalizedFeatures[name] : 0
            );
            
            // Convert to tensor
            const input = tf.tensor2d([featureVector]);
            
            // Make prediction
            const prediction = await this.model.predict(input);
            const probability = prediction.dataSync()[0];
            
            // Dispose tensors to free memory
            input.dispose();
            prediction.dispose();
            
            // Calculate confidence level based on model metrics
            const confidence = this.calculateConfidence(probability);
            
            // Determine risk level
            const riskLevel = this.determineRiskLevel(probability);
            
            return {
                probability,
                confidence,
                riskLevel,
                tornado: probability >= 0.5
            };
        } catch (error) {
            console.error("Error making prediction:", error);
            return null;
        }
    }
    
    /**
     * Calculate confidence level based on probability and model metrics
     */
    calculateConfidence(probability) {
        // Simple confidence metric based on how far from 0.5 the prediction is
        const distanceFromUncertain = Math.abs(probability - 0.5) * 2; // 0 to 1
        
        // Weight by model precision or recall depending on the prediction
        if (probability >= 0.5) {
            // For positive predictions, weight by precision (avoid false positives)
            return distanceFromUncertain * (0.5 + this.metrics.precision * 0.5);
        } else {
            // For negative predictions, weight by recall (avoid false negatives)
            return distanceFromUncertain * (0.5 + this.metrics.recall * 0.5);
        }
    }
    
    /**
     * Determine risk level based on probability
     */
    determineRiskLevel(probability) {
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
     * Train the model on new data
     */
    async trainOnNewData(trainingData) {
        try {
            if (!this.model) {
                console.error("Model not initialized");
                return false;
            }
            
            if (!trainingData || trainingData.length === 0) {
                console.log("No training data provided");
                return false;
            }
            
            console.log(`Training model on ${trainingData.length} new examples`);
            
            // Update feature stats with new data
            this.updateFeatureStats(trainingData);
            
            // Convert training data to tensors
            const { xs, ys } = this.convertToTensors(trainingData);
            
            // Define training parameters
            const batchSize = Math.min(32, trainingData.length);
            const epochs = 5;
            
            // Train the model
            const history = await this.model.fit(xs, ys, {
                batchSize,
                epochs,
                validationSplit: 0.2,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        console.log(`Epoch ${epoch + 1} of ${epochs}:`);
                        console.log(`  Loss: ${logs.loss.toFixed(4)}, Accuracy: ${logs.acc.toFixed(4)}`);
                    }
                }
            });
            
            // Update metrics
            this.calculateMetrics(trainingData);
            this.metrics.trainingExamples += trainingData.length;
            this.metrics.lastTrainingDate = new Date().toISOString();
            
            // Save metrics to localStorage
            localStorage.setItem('tornadoModelMetrics', JSON.stringify(this.metrics));
            
            // Save model to IndexedDB
            await this.saveModel();
            
            // Dispose tensors to free memory
            xs.dispose();
            ys.dispose();
            
            console.log("Model training complete");
            return true;
        } catch (error) {
            console.error("Error training model on new data:", error);
            return false;
        }
    }
    
    /**
     * Update feature stats with new data
     */
    updateFeatureStats(newData) {
        try {
            // If no existing stats, calculate from scratch
            if (Object.keys(this.featureStats).length === 0) {
                return this.calculateFeatureStats(newData);
            }
            
            // Get current total examples
            const currentTotal = this.metrics.trainingExamples || 0;
            const newTotal = currentTotal + newData.length;
            
            // For each feature, update stats
            this.featureNames.forEach(feature => {
                const values = newData.map(example => example.features[feature]);
                
                const newMin = Math.min(...values);
                const newMax = Math.max(...values);
                
                // Update min and max
                if (!this.featureStats[feature]) {
                    this.featureStats[feature] = { min: newMin, max: newMax, mean: 0, std: 1 };
                } else {
                    this.featureStats[feature].min = Math.min(this.featureStats[feature].min, newMin);
                    this.featureStats[feature].max = Math.max(this.featureStats[feature].max, newMax);
                }
                
                // Update mean (weighted average of old and new)
                const oldMean = this.featureStats[feature].mean || 0;
                const newMean = values.reduce((a, b) => a + b, 0) / values.length;
                
                const weightedMean = (oldMean * currentTotal + newMean * newData.length) / newTotal;
                this.featureStats[feature].mean = weightedMean;
                
                // Updating std is complex and would require storing all previous data
                // For simplicity, we'll just use a heuristic approach
                const oldStd = this.featureStats[feature].std || 1;
                const newValues = values.map(v => (v - newMean) ** 2);
                const newStd = Math.sqrt(newValues.reduce((a, b) => a + b, 0) / values.length);
                
                // Weighted average of std (approximation)
                const weightedStd = (oldStd * currentTotal + newStd * newData.length) / newTotal;
                this.featureStats[feature].std = weightedStd > 0 ? weightedStd : 1;
            });
            
            // Save updated stats to localStorage
            localStorage.setItem('tornadoFeatureStats', JSON.stringify(this.featureStats));
            
            return true;
        } catch (error) {
            console.error("Error updating feature stats:", error);
            return false;
        }
    }
    
    /**
     * Calculate model performance metrics on a dataset
     */
    calculateMetrics(evaluationData) {
        try {
            if (!evaluationData || evaluationData.length === 0) {
                return;
            }
            
            // Make predictions on each example
            const predictions = [];
            const actuals = [];
            
            for (const example of evaluationData) {
                // Get features
                const normalizedFeatures = this.normalizeFeatures(example.features);
                
                // Create feature vector in the correct order
                const featureVector = this.featureNames.map(name => normalizedFeatures[name]);
                
                // Convert to tensor
                const input = tf.tensor2d([featureVector]);
                
                // Make prediction
                const prediction = this.model.predict(input);
                const probability = prediction.dataSync()[0];
                
                predictions.push(probability >= 0.5);
                actuals.push(example.tornado);
                
                // Dispose tensors
                input.dispose();
                prediction.dispose();
            }
            
            // Calculate confusion matrix
            let truePositives = 0;
            let falsePositives = 0;
            let trueNegatives = 0;
            let falseNegatives = 0;
            
            for (let i = 0; i < predictions.length; i++) {
                if (predictions[i] && actuals[i]) {
                    truePositives++;
                } else if (predictions[i] && !actuals[i]) {
                    falsePositives++;
                } else if (!predictions[i] && !actuals[i]) {
                    trueNegatives++;
                } else if (!predictions[i] && actuals[i]) {
                    falseNegatives++;
                }
            }
            
            // Calculate metrics
            this.metrics.accuracy = (truePositives + trueNegatives) / evaluationData.length;
            this.metrics.precision = truePositives / (truePositives + falsePositives) || 0;
            this.metrics.recall = truePositives / (truePositives + falseNegatives) || 0;
            
            // F1 score
            this.metrics.f1Score = 2 * (this.metrics.precision * this.metrics.recall) / 
                (this.metrics.precision + this.metrics.recall) || 0;
            
            return this.metrics;
        } catch (error) {
            console.error("Error calculating metrics:", error);
            return this.metrics;
        }
    }
    
    /**
     * Update model with feedback from verification
     */
    async updateWithFeedback(prediction, actualTornado) {
        try {
            // Create a training example from the prediction and actual outcome
            const trainingExample = {
                features: prediction.features,
                tornado: actualTornado,
                date: new Date().toISOString()
            };
            
            // Add to data collector if available
            if (window.tornadoDataCollector) {
                window.tornadoDataCollector.addTornadoObservation(prediction.features, actualTornado);
            }
            
            // Train on this single example
            await this.trainOnNewData([trainingExample]);
            
            console.log(`Model updated with feedback. Actual tornado: ${actualTornado}`);
            return true;
        } catch (error) {
            console.error("Error updating model with feedback:", error);
            return false;
        }
    }
    
    /**
     * Get current model metrics
     */
    getMetrics() {
        return { ...this.metrics };
    }
}

// Create singleton instance
const tornadoPredictionModel = new TornadoPredictionModel();

// Export the model
window.tornadoPredictionModel = tornadoPredictionModel; 
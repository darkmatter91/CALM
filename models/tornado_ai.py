import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
import io
import requests
import pandas as pd

logger = logging.getLogger(__name__)

class TornadoAI:
    """Deep learning model for tornado prediction using radar imagery and weather data."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the TornadoAI model.
        
        Args:
            model_path: Optional path to a saved model.
        """
        self.model = None
        self.radar_model = None
        self.weather_model = None
        self.combined_model = None
        self.weather_scaler = StandardScaler()
        self.model_path = model_path or 'models/saved/tornado_ai'
        self.radar_image_size = (224, 224)  # Standard size for pretrained models
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Load model if it exists
        self._load_model()

    def _load_model(self):
        """Load the model from disk if it exists."""
        try:
            if os.path.exists(f"{self.model_path}_combined.h5"):
                logger.info("Loading saved TornadoAI model...")
                self.combined_model = load_model(f"{self.model_path}_combined.h5")
                self.model = self.combined_model
                
                # Load component models if they exist
                if os.path.exists(f"{self.model_path}_radar.h5"):
                    self.radar_model = load_model(f"{self.model_path}_radar.h5")
                if os.path.exists(f"{self.model_path}_weather.h5"):
                    self.weather_model = load_model(f"{self.model_path}_weather.h5")
                
                # Load scaler if it exists
                if os.path.exists(f"{self.model_path}_scaler.json"):
                    with open(f"{self.model_path}_scaler.json", 'r') as f:
                        scaler_data = json.load(f)
                        self.weather_scaler.mean_ = np.array(scaler_data['mean'])
                        self.weather_scaler.scale_ = np.array(scaler_data['scale'])
                        self.weather_scaler.var_ = np.array(scaler_data['var'])
                        self.weather_scaler.n_features_in_ = scaler_data['n_features_in']
                
                logger.info("Model loaded successfully")
                return True
            else:
                logger.info("No saved model found. Creating new model.")
                self._build_model()
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Creating new model instead.")
            self._build_model()
            return False

    def _build_model(self):
        """Build the deep learning model architecture."""
        logger.info("Building TornadoAI model...")
        
        # 1. Radar Image Processing Model (CNN-based)
        base_model = applications.ResNet50V2(
            include_top=False, 
            weights='imagenet',
            input_shape=(*self.radar_image_size, 3)
        )
        
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        # Add custom layers on top
        radar_input = layers.Input(shape=(*self.radar_image_size, 3))
        x = base_model(radar_input)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        radar_output = layers.Dense(128, activation='relu')(x)
        
        # Create the radar model
        self.radar_model = Model(inputs=radar_input, outputs=radar_output, name="radar_model")
        
        # 2. Weather Data Processing Model (Dense Neural Network)
        # Input features for weather data
        # [CAPE, Helicity, Wind Shear, Temperature, Humidity, Pressure, etc.]
        weather_input = layers.Input(shape=(10,), name="weather_input")
        y = layers.Dense(64, activation='relu')(weather_input)
        y = layers.BatchNormalization()(y)
        y = layers.Dense(32, activation='relu')(y)
        y = layers.Dropout(0.2)(y)
        weather_output = layers.Dense(32, activation='relu')(y)
        
        # Create the weather model
        self.weather_model = Model(inputs=weather_input, outputs=weather_output, name="weather_model")
        
        # 3. Combined Model
        combined_input = [radar_input, weather_input]
        combined = layers.Concatenate()([self.radar_model.output, self.weather_model.output])
        z = layers.Dense(128, activation='relu')(combined)
        z = layers.BatchNormalization()(z)
        z = layers.Dense(64, activation='relu')(z)
        z = layers.Dropout(0.3)(z)
        
        # Output layers for different predictions
        tornado_probability = layers.Dense(1, activation='sigmoid', name='tornado_probability')(z)
        severity_output = layers.Dense(4, activation='softmax', name='severity')(z)  # [Low, Moderate, High, Extreme]
        
        # Create the combined model
        self.combined_model = Model(
            inputs=combined_input,
            outputs=[tornado_probability, severity_output],
            name="tornado_prediction_model"
        )
        
        # Compile the model
        self.combined_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'tornado_probability': 'binary_crossentropy',
                'severity': 'categorical_crossentropy'
            },
            metrics={
                'tornado_probability': ['accuracy', tf.keras.metrics.AUC()],
                'severity': ['accuracy']
            }
        )
        
        # Set the main model
        self.model = self.combined_model
        logger.info("TornadoAI model built successfully")

    def preprocess_radar_image(self, radar_data: np.ndarray) -> np.ndarray:
        """Preprocess radar image data for the model.
        
        Args:
            radar_data: Raw radar data or image array
            
        Returns:
            Preprocessed image array ready for model input
        """
        try:
            # Special handling for empty arrays (from failed downloads)
            if isinstance(radar_data, np.ndarray) and np.all(radar_data == 0):
                # Return a properly formatted empty image with correct dimensions
                return np.zeros((1, *self.radar_image_size, 3), dtype=np.float32)
                
            # Handle different input types
            if isinstance(radar_data, str):
                # URL or file path
                if radar_data.startswith(('http://', 'https://')):
                    response = requests.get(radar_data)
                    img = Image.open(io.BytesIO(response.content))
                else:
                    img = Image.open(radar_data)
                
                # Convert to RGB if the image has alpha channel (RGBA)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                img = img.resize(self.radar_image_size)
                img_array = img_to_array(img)
            elif isinstance(radar_data, Image.Image):
                # PIL Image
                if radar_data.mode == 'RGBA':
                    radar_data = radar_data.convert('RGB')
                img = radar_data.resize(self.radar_image_size)
                img_array = img_to_array(img)
            elif isinstance(radar_data, np.ndarray):
                # NumPy array
                # Check if we have an RGBA array (4 channels) and convert to RGB (3 channels)
                if len(radar_data.shape) == 3 and radar_data.shape[2] == 4:
                    # Convert RGBA to RGB by dropping the alpha channel
                    radar_data = radar_data[:, :, :3]
                
                # If single-channel, convert to RGB
                if len(radar_data.shape) == 3 and radar_data.shape[2] == 1:
                    radar_data = np.repeat(radar_data, 3, axis=2)
                elif radar_data.shape[0] == 1 and len(radar_data.shape) == 3:
                    # Single channel - convert to RGB
                    radar_data = np.repeat(radar_data, 3, axis=0)
                
                # Resize if needed
                if radar_data.shape[1:3] != self.radar_image_size or len(radar_data.shape) < 3 or radar_data.shape[2] != 3:
                    try:
                        # Ensure array is uint8 for PIL
                        radar_data_uint8 = np.uint8(np.clip(radar_data, 0, 255))
                        
                        # Convert to PIL Image for easy resizing
                        if len(radar_data.shape) == 2:  # If 2D array, create grayscale image
                            img = Image.fromarray(radar_data_uint8, mode='L').convert('RGB')
                        else:
                            img = Image.fromarray(radar_data_uint8)
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                        
                        img = img.resize(self.radar_image_size)
                        img_array = img_to_array(img)
                    except Exception as e:
                        logger.error(f"Error resizing array: {e}")
                        return np.zeros((1, *self.radar_image_size, 3), dtype=np.float32)
                else:
                    img_array = radar_data
                    
                # Ensure we have exactly 3 channels
                if len(img_array.shape) == 3 and img_array.shape[2] != 3:
                    if img_array.shape[2] == 4:  # RGBA
                        img_array = img_array[:, :, :3]
                    elif img_array.shape[2] == 1:  # Grayscale
                        img_array = np.repeat(img_array, 3, axis=2)
            else:
                logger.error(f"Unsupported radar data type: {type(radar_data)}")
                return np.zeros((1, *self.radar_image_size, 3), dtype=np.float32)
            
            # Final check to ensure we have exactly 3 channels
            if len(img_array.shape) == 3 and img_array.shape[2] != 3:
                logger.error(f"Invalid image shape after processing: {img_array.shape}")
                return np.zeros((1, *self.radar_image_size, 3), dtype=np.float32)
            
            # Normalize
            img_array = img_array / 255.0
            
            # Ensure correct shape (batch dimension)
            if len(img_array.shape) == 3:
                img_array = np.expand_dims(img_array, axis=0)
                
            # Ensure float32 data type
            img_array = img_array.astype(np.float32)
            
            # Final verification of shape
            if img_array.shape[3] != 3:
                logger.error(f"Invalid final image shape: {img_array.shape}, expected 3 channels")
                return np.zeros((1, *self.radar_image_size, 3), dtype=np.float32)
                
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocessing radar image: {e}")
            # Return empty array with correct shape and dtype
            return np.zeros((1, *self.radar_image_size, 3), dtype=np.float32)

    def preprocess_weather_data(self, weather_metrics: Dict[str, Any]) -> np.ndarray:
        """Preprocess weather data for the model.
        
        Args:
            weather_metrics: Dictionary of weather metrics
            
        Returns:
            Preprocessed weather data array ready for model input
        """
        try:
            # Extract relevant features and ensure consistent order
            # These should match the expected input features for the model
            features = [
                weather_metrics.get('cape', 0),                          # CAPE value
                weather_metrics.get('helicity', 0),                      # Helicity
                weather_metrics.get('wind_shear', 0),                    # Wind shear (extract numeric value)
                weather_metrics.get('storm_motion', 0),                  # Storm motion speed
                weather_metrics.get('temperature', 70),                  # Temperature (F)
                weather_metrics.get('dewpoint', 50),                     # Dewpoint (F) 
                weather_metrics.get('humidity', 50),                     # Relative humidity (%)
                weather_metrics.get('pressure', 1013.25),                # Pressure (hPa)
                weather_metrics.get('precipitable_water', 25),           # Precipitable water (mm)
                weather_metrics.get('lapse_rate', 6.5)                   # Lapse rate (C/km)
            ]
            
            # Convert string or other types to float
            for i, feature in enumerate(features):
                if isinstance(feature, str):
                    # Extract numeric value from string like "30 knots"
                    numeric_value = ''.join(filter(str.isdigit, feature))
                    features[i] = float(numeric_value) if numeric_value else 0
                elif not isinstance(feature, (int, float)):
                    features[i] = 0
            
            # Convert to numpy array
            features_array = np.array(features, dtype=np.float32).reshape(1, -1)
            
            # Scale the features if the model has been trained
            if hasattr(self.weather_scaler, 'mean_'):
                features_array = self.weather_scaler.transform(features_array)
            
            return features_array
            
        except Exception as e:
            logger.error(f"Error preprocessing weather data: {e}")
            # Return zero array with correct shape on error
            return np.zeros((1, 10), dtype=np.float32)

    def predict(self, radar_data: Optional[np.ndarray] = None, 
                weather_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a prediction using the model.
        
        Args:
            radar_data: Optional radar data. If None, prediction will be based on weather data only.
            weather_data: Optional weather metrics. If None, prediction will be based on radar data only.
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Check if we can make a prediction
            if self.model is None:
                logger.error("Model not loaded or trained")
                return {
                    'status': 'error',
                    'message': 'Model not loaded or trained'
                }
                
            # If both data sources are None, return error
            if radar_data is None and weather_data is None:
                logger.error("No data provided for prediction")
                return {
                    'status': 'error',
                    'message': 'No data provided for prediction'
                }
            
            # Preprocess radar data (or use fallback empty image)
            if radar_data is not None:
                processed_radar = self.preprocess_radar_image(radar_data)
            else:
                # Create an empty radar image
                processed_radar = np.zeros((1, *self.radar_image_size, 3), dtype=np.float32)
                logger.warning("Using empty radar data for prediction")
            
            # Preprocess weather data (or use fallback values)
            if weather_data is not None:
                processed_weather = self.preprocess_weather_data(weather_data)
            else:
                # Create empty weather data with zeros
                processed_weather = np.zeros((1, 10), dtype=np.float32)
                logger.warning("Using empty weather data for prediction")
            
            # Make prediction
            tornado_prob, severity = self.model.predict([processed_radar, processed_weather], verbose=0)
            
            # Interpret results
            tornado_probability = float(tornado_prob[0][0])
            severity_index = np.argmax(severity[0])
            severity_mapping = {0: 'low', 1: 'moderate', 2: 'high', 3: 'extreme'}
            severity_label = severity_mapping.get(severity_index, 'unknown')
            
            # Calculate confidence based on the strength of the prediction
            confidence = float(np.max(severity[0]))
            
            # Apply meteorological constraints to ensure realistic predictions
            # Extract key parameters from weather_data
            cape = weather_data.get('cape', 0) if weather_data else 0
            helicity = weather_data.get('helicity', 0) if weather_data else 0
            wind_shear = weather_data.get('wind_shear', 0) if weather_data else 0
            
            # Convert wind_shear to a number if it's a string
            if isinstance(wind_shear, str):
                wind_shear = int(''.join(filter(str.isdigit, wind_shear))) if any(c.isdigit() for c in wind_shear) else 0
            
            # IMPORTANT OVERRIDE: For locations with high helicity values,
            # Force a higher tornado probability regardless of model output
            # This ensures we don't miss actual tornado conditions
            if helicity >= 150 and cape >= 1200 and wind_shear >= 20:
                # This is a realistic tornadic environment - override model's low values
                # But make it more nuanced and less aggressive
                
                # Calculate a balanced probability that varies based on actual conditions
                # Instead of forcing to max values
                h_factor = min(1.0, helicity / 300)  # Scale factor for helicity (max at 300)
                c_factor = min(1.0, cape / 3000)     # Scale factor for CAPE (max at 3000)
                s_factor = min(1.0, wind_shear / 50) # Scale factor for wind shear (max at 50)
                
                # Use multiplicative formula but with square root to moderate extremes
                raw_prob = (h_factor * c_factor * s_factor) ** 0.5
                
                # Scale to a more reasonable range (0.3 to 0.8) rather than forcing to 0.95
                forced_probability = 0.3 + (raw_prob * 0.5)
                
                # Only override if our calculated value is higher than model's prediction
                if forced_probability > tornado_probability:
                    tornado_probability = forced_probability
                    
                    # Choose severity level based on actual values, not forcing to extreme
                    if helicity >= 250 and cape >= 2500 and wind_shear >= 40:
                        severity_index = max(severity_index, 3)  # Force extreme risk only for truly extreme conditions
                    elif helicity >= 200 and cape >= 2000:
                        severity_index = max(severity_index, 2)  # Force high risk for strong conditions
                    elif helicity >= 150:
                        severity_index = max(severity_index, 1)  # Force moderate risk for moderate conditions
                    
                    # Log that we're applying a forced override
                    logger.info(f"OVERRIDE: Applied balanced probability {forced_probability:.2f} due to helicity={helicity}, CAPE={cape}, wind_shear={wind_shear}")
                else:
                    logger.info(f"Model prediction of {tornado_probability:.2f} retained (higher than calculated {forced_probability:.2f})")
            
            # Apply meteorological constraints based on known tornado formation conditions
            # These thresholds are based on meteorological research
            
            # CAPE constraints - MAKE MORE CONSERVATIVE
            if cape < 500:  # Increased from 300
                tornado_probability = min(tornado_probability, 0.25)  # Reduced from 0.3
                severity_index = min(severity_index, 1)  # Force low severity
            elif cape < 1000:  # Increased from 800
                tornado_probability = min(tornado_probability, 0.4)  # Reduced from 0.5
                severity_index = min(severity_index, 2)  # Cap at moderate severity
            
            # Helicity constraints - MAKE MORE CONSERVATIVE
            if helicity < 50:  # Increased from 30
                tornado_probability = min(tornado_probability, 0.15)  # Reduced from 0.2
                severity_index = min(severity_index, 1)  # Force low severity
            elif helicity < 150:  # Increased from 100
                tornado_probability = min(tornado_probability, 0.35)  # Reduced from 0.4
                severity_index = min(severity_index, 2)  # Cap at moderate severity
            
            # Wind shear constraints - MAKE MORE CONSERVATIVE
            if wind_shear < 15:  # Increased from 10
                tornado_probability = min(tornado_probability, 0.2)  # Reduced from 0.3
                severity_index = min(severity_index, 1)  # Force low severity
            elif wind_shear < 25:  # Increased from 20
                tornado_probability = min(tornado_probability, 0.5)  # Reduced from 0.6
                severity_index = min(severity_index, 2)  # Cap at moderate severity
            
            # Combined constraint - MAKE MORE CONSERVATIVE
            if cape < 1500 or helicity < 150 or wind_shear < 25:  # Increased thresholds
                tornado_probability = min(tornado_probability, 0.7)  # Reduced from 0.8
                severity_index = min(severity_index, 2)  # Cap at moderate
            
            # Extreme risk - MAKE MORE CONSERVATIVE
            if cape < 2500 or helicity < 250 or wind_shear < 35:  # Increased from 2000/200/30
                severity_index = min(severity_index, 2)  # Cap at high (was 3/extreme)
            
            # Override severity label based on our constraints
            severity_label = severity_mapping.get(severity_index, 'unknown')
            
            # Log the adjustment
            logger.info(f"Adjusted prediction based on meteorological constraints: " +
                     f"CAPE={cape}, Helicity={helicity}, Wind Shear={wind_shear}, " +
                     f"Final probability={tornado_probability}, Final severity={severity_label}")
            
            # Return prediction
            return {
                'status': 'success',
                'probability': tornado_probability,
                'severity': severity_label.upper(),
                'confidence': confidence,
                'raw_output': {
                    'tornado_prob': float(tornado_prob[0][0]),
                    'severity_scores': [float(s) for s in severity[0]],
                    'cape': cape,
                    'helicity': helicity,
                    'wind_shear': wind_shear
                }
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {
                'status': 'error',
                'message': f'Error making prediction: {str(e)}',
                'probability': 0.1,  # Default low probability
                'severity': 'LOW',   # Default low severity
                'confidence': 0.5    # Default moderate confidence
            }

    def train(self, training_data: Dict[str, Any], validation_split: float = 0.2, 
              epochs: int = 50, batch_size: int = 32) -> Dict[str, Any]:
        """Train the model with historical data.
        
        Args:
            training_data: Dictionary containing training data
                - radar_images: List of radar images or file paths
                - weather_metrics: List of weather metric dictionaries
                - labels: List of dictionaries with 'tornado' (0/1) and 'severity' (0-3)
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary containing training results and metrics
        """
        try:
            logger.info(f"Starting model training with {len(training_data.get('labels', []))} samples")
            
            # Extract training data
            radar_images = training_data.get('radar_images', [])
            weather_metrics = training_data.get('weather_metrics', [])
            labels = training_data.get('labels', [])
            
            if not radar_images or not weather_metrics or not labels:
                logger.error("Missing required training data")
                return {'status': 'error', 'message': 'Missing required training data'}
            
            if not (len(radar_images) == len(weather_metrics) == len(labels)):
                logger.error("Training data arrays must have the same length")
                return {'status': 'error', 'message': 'Training data arrays must have the same length'}
            
            # Preprocess radar images
            logger.info("Preprocessing radar images...")
            processed_radar = []
            for image in radar_images:
                processed_radar.append(self.preprocess_radar_image(image)[0])  # Remove batch dimension
            radar_array = np.array(processed_radar)
            
            # Preprocess weather data
            logger.info("Preprocessing weather data...")
            weather_features = []
            for metrics in weather_metrics:
                # Extract features to match model input
                features = [
                    metrics.get('cape', 0),
                    metrics.get('helicity', 0),
                    metrics.get('wind_shear', 0),
                    metrics.get('storm_motion', 0),
                    metrics.get('temperature', 70),
                    metrics.get('dewpoint', 50),
                    metrics.get('humidity', 50),
                    metrics.get('pressure', 1013.25),
                    metrics.get('precipitable_water', 25),
                    metrics.get('lapse_rate', 6.5)
                ]
                weather_features.append(features)
            
            weather_array = np.array(weather_features, dtype=np.float32)
            
            # Fit the scaler to the weather data
            self.weather_scaler.fit(weather_array)
            weather_array_scaled = self.weather_scaler.transform(weather_array)
            
            # Prepare labels
            tornado_labels = np.array([label.get('tornado', 0) for label in labels])
            
            # Convert severity categories to one-hot encoding
            severity_labels = np.array([label.get('severity', 0) for label in labels])
            severity_one_hot = tf.keras.utils.to_categorical(severity_labels, num_classes=4)
            
            # Train-validation split
            indices = np.random.permutation(len(labels))
            split_idx = int(len(indices) * (1 - validation_split))
            train_idx, val_idx = indices[:split_idx], indices[split_idx:]
            
            # Training data
            train_radar = radar_array[train_idx]
            train_weather = weather_array_scaled[train_idx]
            train_tornado = tornado_labels[train_idx]
            train_severity = severity_one_hot[train_idx]
            
            # Validation data
            val_radar = radar_array[val_idx]
            val_weather = weather_array_scaled[val_idx]
            val_tornado = tornado_labels[val_idx]
            val_severity = severity_one_hot[val_idx]
            
            # Setup callbacks
            callbacks = [
                ModelCheckpoint(
                    filepath=f"{self.model_path}_best.h5",
                    monitor='val_tornado_probability_accuracy',
                    save_best_only=True,
                    mode='max'
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
            
            # Train the model
            logger.info(f"Training model with {len(train_idx)} samples, validating with {len(val_idx)} samples")
            history = self.model.fit(
                [train_radar, train_weather],
                [train_tornado, train_severity],
                validation_data=([val_radar, val_weather], [val_tornado, val_severity]),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks
            )
            
            # Save the trained model
            self._save_model()
            
            # Return training metrics
            return {
                'status': 'success',
                'message': 'Model trained successfully',
                'metrics': {
                    'accuracy': history.history['tornado_probability_accuracy'][-1],
                    'val_accuracy': history.history['val_tornado_probability_accuracy'][-1],
                    'severity_accuracy': history.history['severity_accuracy'][-1],
                    'val_severity_accuracy': history.history['val_severity_accuracy'][-1]
                },
                'epochs_completed': len(history.history['loss'])
            }
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return {
                'status': 'error',
                'message': f'Error training model: {str(e)}'
            }

    def _save_model(self):
        """Save the model to disk."""
        try:
            # Save the combined model
            self.combined_model.save(f"{self.model_path}_combined.h5")
            
            # Save component models
            if self.radar_model:
                self.radar_model.save(f"{self.model_path}_radar.h5")
            if self.weather_model:
                self.weather_model.save(f"{self.model_path}_weather.h5")
            
            # Save the scaler parameters
            if hasattr(self.weather_scaler, 'mean_'):
                scaler_data = {
                    'mean': self.weather_scaler.mean_.tolist(),
                    'scale': self.weather_scaler.scale_.tolist(),
                    'var': self.weather_scaler.var_.tolist(),
                    'n_features_in': self.weather_scaler.n_features_in_
                }
                with open(f"{self.model_path}_scaler.json", 'w') as f:
                    json.dump(scaler_data, f)
            
            logger.info(f"Model saved to {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def download_radar_image(self, lat: float, lon: float, zoom: int = 8) -> np.ndarray:
        """Download radar image for the specified location.
        
        Args:
            lat: Latitude
            lon: Longitude
            zoom: Zoom level (1-15)
            
        Returns:
            Numpy array of the radar image
        """
        try:
            # First fetch the RainViewer API data to get available radar frames
            api_url = "https://api.rainviewer.com/public/weather-maps.json"
            api_response = requests.get(api_url, timeout=10)
            api_response.raise_for_status()
            api_data = api_response.json()
            
            # Get the host and the most recent radar frame path
            host = api_data.get('host')
            radar_frames = api_data.get('radar', {}).get('past', [])
            
            if not host or not radar_frames:
                logger.error("No radar data available from RainViewer API")
                return np.zeros((*self.radar_image_size, 3))
            
            # Get the most recent frame
            latest_frame = radar_frames[-1]
            path = latest_frame.get('path')
            
            if not path:
                logger.error("Invalid radar frame data from RainViewer API")
                return np.zeros((*self.radar_image_size, 3))
            
            # Construct URL according to RainViewer API format
            # Format: {host}{path}/{size}/{z}/{lat}/{lon}/{color}/{options}.png
            size = 256  # 256 or 512
            color = 1   # Color scheme (1 is default)
            options = "1_1"  # smooth=1, snow=1
            
            url = f"{host}{path}/{size}/{zoom}/{lat}/{lon}/{color}/{options}.png"
            logger.info(f"Downloading radar image from: {url}")
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Open the image and convert to RGB if it's RGBA
            img = Image.open(io.BytesIO(response.content))
            if img.mode == 'RGBA':
                img = img.convert('RGB')
                
            # Resize to match model input requirements
            if img.size != self.radar_image_size:
                img = img.resize(self.radar_image_size)
                
            # Convert to numpy array
            img_array = img_to_array(img)
            
            # Verify the shape has 3 channels (RGB)
            if img_array.shape[2] != 3:
                logger.warning(f"Image has {img_array.shape[2]} channels, converting to RGB")
                if img_array.shape[2] == 4:  # RGBA
                    img_array = img_array[:, :, :3]
                elif img_array.shape[2] == 1:  # Grayscale
                    img_array = np.repeat(img_array, 3, axis=2)
            
            return img_array
        except requests.exceptions.HTTPError as e:
            logger.error(f"Error downloading radar image: {e}")
            # Return empty array with correct shape (3 channels)
            return np.zeros((*self.radar_image_size, 3))
        except Exception as e:
            logger.error(f"Error processing radar image: {e}")
            # Return empty array with correct shape (3 channels)
            return np.zeros((*self.radar_image_size, 3))

    def generate_training_data(self, nws_data_path: str, 
                              save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate training data from NWS historical data.
        
        Args:
            nws_data_path: Path to NWS historical data CSV
            save_path: Optional path to save processed data
            
        Returns:
            Dictionary containing training data or status
        """
        try:
            if not os.path.exists(nws_data_path):
                return {'status': 'error', 'message': f'Data file not found: {nws_data_path}'}
            
            # Load historical tornado data
            df = pd.read_csv(nws_data_path)
            logger.info(f"Loaded {len(df)} records from {nws_data_path}")
            
            # Process and prepare training data
            # This is a placeholder - in a real implementation, you would:
            # 1. Load historical tornado reports
            # 2. Match with radar imagery from the time
            # 3. Extract weather parameters
            # 4. Create labeled training samples
            
            # For now, return a status message
            return {
                'status': 'not_implemented',
                'message': 'Training data generation from historical data is not yet implemented'
            }
            
        except Exception as e:
            logger.error(f"Error generating training data: {e}")
            return {
                'status': 'error',
                'message': f'Error generating training data: {str(e)}'
            } 
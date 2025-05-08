import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import random
import os

class WeatherPredictor:
    """A simplified class for predicting extreme weather events."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the weather predictor.
        
        Args:
            model_path: Optional path to a saved model (ignored in this simplified version).
        """
        pass
        
    def predict(self, location: Dict[str, float], timestamp: str) -> Dict[str, Any]:
        """Make weather predictions for a given location and time.
        
        Args:
            location: Dictionary containing 'latitude' and 'longitude'
            timestamp: ISO format timestamp string
            
        Returns:
            Dictionary containing prediction results:
                - probability: Float between 0 and 1
                - severity: String indicating severity level
                - features: Dictionary of input features used
        """
        try:
            # Convert timestamp to datetime
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00') if timestamp.endswith('Z') else timestamp)
            
            # Generate a random probability based on location and time
            # This is just a simplified mock implementation
            lat = location.get('latitude', 0)
            lon = location.get('longitude', 0)
            
            # Use timestamp components to add some variability
            month_factor = dt.month / 12  # Higher in summer months for US
            hour_factor = abs(dt.hour - 12) / 12  # Higher in afternoon
            
            # Generate a base random probability
            base_probability = random.random() * 0.7
            
            # Adjust based on location (higher for tornado alley)
            if 30 <= lat <= 45 and -100 <= lon <= -85:
                # Tornado alley area - higher probability
                location_factor = 0.3
            else:
                location_factor = 0.1
                
            # Calculate final probability
            probability = min(0.95, base_probability + location_factor + (month_factor * 0.2) + (hour_factor * 0.1))
            
            # Determine severity
            severity = self._get_severity_level(probability)
            
            return {
                'probability': probability,
                'severity': severity,
                'features': {
                    'latitude': location.get('latitude', 0),
                    'longitude': location.get('longitude', 0),
                    'timestamp': timestamp
                }
            }
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            # Return default values in case of error
            return {
                'probability': 0.1,
                'severity': 'LOW',
                'features': {
                    'latitude': location.get('latitude', 0),
                    'longitude': location.get('longitude', 0),
                    'timestamp': timestamp
                }
            }
            
    def _get_severity_level(self, probability: float) -> str:
        """Convert probability to severity level.
        
        Args:
            probability: Float between 0 and 1
            
        Returns:
            String indicating severity level
        """
        if probability < 0.3:
            return 'LOW'
        elif probability < 0.6:
            return 'MODERATE'
        elif probability < 0.8:
            return 'HIGH'
        else:
            return 'EXTREME' 
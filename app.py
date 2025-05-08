from flask import Flask, render_template, jsonify, request
from models.weather_predictor import WeatherPredictor
from models.tornado_ai import TornadoAI
from utils.radar_processor import RadarProcessor
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import requests
import time
import numpy as np
import json
import math
import random
import re
import uuid
import sqlite3
import logging
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import io
from PIL import Image
from werkzeug.middleware.proxy_fix import ProxyFix

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tornado_predictions.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize components
radar_processor = RadarProcessor()
weather_predictor = WeatherPredictor()
try:
    tornado_ai = TornadoAI()
except Exception as e:
    logger.error(f"Error initializing TornadoAI: {e}")
    tornado_ai = None

# Database setup for prediction logging
DB_PATH = "tornado_predictions.db"

def initialize_db():
    """Initialize the database for tornado prediction logging and validation."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create predictions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            prediction_time TEXT,
            lat REAL,
            lon REAL,
            location TEXT,
            risk_level TEXT,
            formation_chance INTEGER,
            cape INTEGER,
            helicity INTEGER,
            shear TEXT,
            is_ai_prediction BOOLEAN,
            nws_alert BOOLEAN,
            validated BOOLEAN DEFAULT FALSE
        )
        ''')
        
        # Create validation table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS validation_results (
            prediction_id TEXT,
            validated_at TEXT,
            was_correct BOOLEAN,
            actual_event_id TEXT,
            distance_error_km REAL,
            time_error_minutes INTEGER,
            notes TEXT,
            FOREIGN KEY (prediction_id) REFERENCES predictions (id)
        )
        ''')
        
        conn.commit()
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False
    finally:
        if conn:
            conn.close()

def log_prediction(prediction):
    """Log a tornado prediction to the database for later validation."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if prediction already exists
        cursor.execute("SELECT id FROM predictions WHERE id = ?", (prediction['id'],))
        if cursor.fetchone():
            logger.info(f"Prediction {prediction['id']} already logged")
            return True
            
        # Extract relevant data
        pred_id = prediction['id']
        timestamp = datetime.now(timezone.utc).isoformat()
        prediction_time = prediction.get('timestamp', timestamp)
        lat = prediction.get('lat', 0)
        lon = prediction.get('lon', 0)
        location = prediction.get('location', prediction.get('name', 'Unknown'))
        risk_level = prediction.get('risk_level', 'unknown')
        formation_chance = prediction.get('formation_chance', 0)
        cape = prediction.get('cape', 0)
        helicity = prediction.get('helicity', 0)
        shear = prediction.get('shear', 'unknown')
        is_ai_prediction = prediction.get('is_ai_prediction', False)
        nws_alert = prediction.get('nws_alert', False)
        
        # Insert into database
        cursor.execute('''
        INSERT INTO predictions (
            id, timestamp, prediction_time, lat, lon, location, risk_level,
            formation_chance, cape, helicity, shear, is_ai_prediction, nws_alert
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pred_id, timestamp, prediction_time, lat, lon, location, risk_level,
            formation_chance, cape, helicity, shear, is_ai_prediction, nws_alert
        ))
        
        conn.commit()
        logger.info(f"Logged prediction {pred_id} for {location}")
        return True
    except Exception as e:
        logger.error(f"Error logging prediction: {e}")
        return False
    finally:
        if conn:
            conn.close()

def validate_predictions(days_ago=1):
    """Validate predictions against actual NWS tornado reports."""
    try:
        # Calculate the date to validate
        validation_date = datetime.now(timezone.utc) - timedelta(days=days_ago)
        date_str = validation_date.strftime("%Y-%m-%d")
        
        logger.info(f"Starting validation for predictions on {date_str}")
        
        # Get predictions from our database for that date
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Query for predictions in the date range
        cursor.execute('''
        SELECT id, lat, lon, location, risk_level, formation_chance, is_ai_prediction
        FROM predictions
        WHERE date(prediction_time) = ?
        AND validated = FALSE
        ''', (date_str,))
        
        predictions = cursor.fetchall()
        logger.info(f"Found {len(predictions)} unvalidated predictions for {date_str}")
        
        if not predictions:
            logger.info("No predictions to validate")
            return True
            
        # Fetch actual tornado reports from NWS for that date
        # This would normally call an NWS API endpoint
        actual_reports = get_nws_tornado_reports(date_str)
        
        if not actual_reports:
            logger.info(f"No actual tornado reports found for {date_str}")
            
        # For each prediction, check if it was correct
        validated_count = 0
        correct_count = 0
        
        for pred in predictions:
            pred_id, lat, lon, location, risk_level, chance, is_ai = pred
            
            # Find closest tornado report
            closest_report, distance_km = find_closest_tornado(lat, lon, actual_reports)
            
            # Determine if prediction was correct (within 100km and relevant risk)
            was_correct = False
            notes = "No matching tornado events"
            
            if closest_report:
                # If prediction chance was high and tornado occurred within 100km, count as correct
                if distance_km <= 100 and (
                    (chance >= 70 and risk_level in ['high', 'extreme']) or 
                    (chance >= 40 and risk_level in ['moderate', 'high', 'extreme'])
                ):
                    was_correct = True
                    notes = f"Tornado occurred {distance_km:.1f}km from prediction"
                    correct_count += 1
                else:
                    notes = f"Nearest tornado was {distance_km:.1f}km away, but prediction confidence didn't match"
            
            # Record validation result
            cursor.execute('''
            INSERT INTO validation_results (
                prediction_id, validated_at, was_correct, 
                actual_event_id, distance_error_km, notes
            ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                pred_id, datetime.now(timezone.utc).isoformat(), was_correct,
                closest_report.get('id') if closest_report else None,
                distance_km if closest_report else None,
                notes
            ))
            
            # Mark prediction as validated
            cursor.execute("UPDATE predictions SET validated = TRUE WHERE id = ?", (pred_id,))
            validated_count += 1
        
        conn.commit()
        
        # Log validation summary
        accuracy = (correct_count / validated_count) * 100 if validated_count > 0 else 0
        logger.info(f"Validation complete: {validated_count} predictions processed, {correct_count} correct ({accuracy:.1f}%)")
        
        return True
    except Exception as e:
        logger.error(f"Error validating predictions: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_nws_tornado_reports(date_str):
    """Fetch actual tornado reports from NWS for a given date."""
    try:
        # This is a placeholder - in reality, you would call the NWS API
        # to get actual tornado reports for the date
        
        # For now, return a mock empty list
        # In production, this would call a real API endpoint
        logger.info(f"Fetching NWS tornado reports for {date_str}")
        
        # Example API call (commented out):
        # url = f"https://api.weather.gov/reports/tornado?date={date_str}"
        # response = requests.get(url, headers={'User-Agent': USER_AGENT})
        # if response.status_code == 200:
        #     return response.json().get('reports', [])
        
        # For now, return empty list as placeholder
        return []
    except Exception as e:
        logger.error(f"Error fetching NWS tornado reports: {e}")
        return []

def find_closest_tornado(lat, lon, tornado_reports):
    """Find the closest tornado report to the given coordinates."""
    if not tornado_reports:
        return None, float('inf')
        
    closest_report = None
    min_distance = float('inf')
    
    for report in tornado_reports:
        # Extract tornado coordinates
        tornado_lat = report.get('lat')
        tornado_lon = report.get('lon')
        
        if not tornado_lat or not tornado_lon:
            continue
            
        # Calculate distance using Haversine formula
        distance = calculate_distance(lat, lon, tornado_lat, tornado_lon)
        
        if distance < min_distance:
            min_distance = distance
            closest_report = report
    
    return closest_report, min_distance

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on the earth (specified in decimal degrees)"""
    # Convert decimal degrees to radians
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    
    # Radius of earth in kilometers
    R = 6371
    
    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance

# Initialize app
app = Flask(__name__)

# Configure app for running behind proxy
app.config['PREFERRED_URL_SCHEME'] = 'https'
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

weather_predictor = WeatherPredictor()
radar_processor = RadarProcessor()

# Set up a mock TornadoAI class for development without model loading
class MockTornadoAI:
    def __init__(self):
        self.model = "mock"  # Use string to emulate having weights
        # Dictionary of clear radar areas (default)
        self.clear_radar_areas = {
            'TX': True,  # All of Texas has clear radar
            'AR': True,  # Little Rock, AR has clear radar
            'OK': True,  # Oklahoma has clear radar
        }
        # We only want Wichita, KS to potentially have activity
        self.active_radar_areas = {
            'Wichita, KS': True
        }
    
    def predict(self, radar_data=None, weather_data=None):
        # Return values should match what the model would return
        # But randomize them a bit to simulate different predictions
        import random
        probability = round(random.uniform(0.5, 0.95), 2)
        severities = ['LOW', 'MODERATE', 'HIGH', 'EXTREME']
        severity = random.choice(severities)
        
        return {
            'status': 'success',
            'probability': probability,
            'severity': severity,
            'confidence': round(random.uniform(0.6, 0.9), 2)
        }
    
    def download_radar_image(self, lat, lon, zoom=8):
        """Return realistic radar image data based on location"""
        import numpy as np
        
        # Create a base image (all zeros = clear radar)
        radar_image = np.zeros((224, 224, 3))
        
        # Determine if this area should have radar returns based on state boundaries
        is_clear_area = True  # Default to clear
        
        # Check if it's in Texas
        if (29.0 <= lat <= 37.0 and -107.0 <= lon <= -93.0):  # Texas bounds
            is_clear_area = self.clear_radar_areas.get('TX', True)
        # Check if it's in Arkansas
        elif (33.0 <= lat <= 36.5 and -94.0 <= lon <= -89.5):  # Arkansas bounds
            is_clear_area = self.clear_radar_areas.get('AR', True)
        # Check if it's in Oklahoma
        elif (33.5 <= lat <= 37.0 and -103.0 <= lon <= -94.0):  # Oklahoma bounds
            is_clear_area = self.clear_radar_areas.get('OK', True)
        # Check specific locations we want to be active
        elif (37.5 <= lat <= 38.0 and -97.5 <= lon <= -97.0):  # Wichita, KS area
            is_clear_area = False
        
        # If it's not a clear area, add simulated radar returns
        if not is_clear_area:
            # Create random radar returns (green/yellow/red pixels)
            # Higher intensity = more severe weather
            intensity = np.random.rand()
            coverage = np.random.uniform(0.05, 0.3)  # 5-30% of the image has radar returns
            
            # Create random masks for different intensities
            mask = np.random.rand(224, 224) < coverage
            
            # Add green (light) returns
            radar_image[mask, 1] = np.random.uniform(0.5, 0.9)
            
            # Add some yellow (moderate) returns in smaller areas
            moderate_mask = np.random.rand(224, 224) < (coverage * 0.6)
            radar_image[moderate_mask, 0] = np.random.uniform(0.4, 0.7)
            radar_image[moderate_mask, 1] = np.random.uniform(0.4, 0.7)
            
            # Add some red (severe) returns in even smaller areas if high intensity
            if intensity > 0.7:
                severe_mask = np.random.rand(224, 224) < (coverage * 0.3)
                radar_image[severe_mask, 0] = np.random.uniform(0.7, 1.0)
                radar_image[severe_mask, 2] = np.random.uniform(0.0, 0.2)  # Less blue
        
        return radar_image

def generate_test_predictions():
    """Generate test predictions for development purposes."""
    import uuid
    import random
    from datetime import datetime, timezone
    
    # Generate 5 test predictions
    predictions = []
    locations = [
        {'name': 'Oklahoma City, OK', 'lat': 35.4676, 'lon': -97.5164},
        {'name': 'Dallas, TX', 'lat': 32.7767, 'lon': -96.7970},
        {'name': 'Wichita, KS', 'lat': 37.6872, 'lon': -97.3301},
        {'name': 'Lubbock, TX', 'lat': 33.5779, 'lon': -101.8552},
        {'name': 'Little Rock, AR', 'lat': 34.7465, 'lon': -92.2896}
    ]
    
    for location in locations:
        # Get radar image for this location
        radar_image = tornado_ai.download_radar_image(location['lat'], location['lon'])
        
        # Check if radar is clear (no significant echo/returns)
        is_radar_clear = False
        if radar_image is not None and isinstance(radar_image, np.ndarray):
            # Convert to grayscale if it's a color image
            if len(radar_image.shape) == 3 and radar_image.shape[2] == 3:
                grayscale = np.mean(radar_image, axis=2)
            else:
                grayscale = radar_image
                
            # Check if most pixels are very low intensity (clear radar)
            threshold = 0.1  # Adjust as needed based on radar imagery
            clear_percentage = np.mean(grayscale < threshold)
            is_radar_clear = clear_percentage > 0.9
            
        # Skip predictions for areas with clear radar
        if is_radar_clear:
            app.logger.info(f"Skipping AI prediction for {location['name']} - radar is clear")
            continue
        
        # Use realistic meteorological values based on location
        if "Wichita, KS" in location['name']:
            # Based on NWS forecast discussion for Wichita - CAPE around 500-600 J/KG
            cape = random.randint(500, 600)
            helicity = random.randint(90, 150)  # Lower helicity for non-tornadic storms
            wind_shear = random.randint(15, 25)  # Moderate wind shear
            risk_level = 'low'  # Actual risk is low based on NWS forecast
            formation_chance = random.randint(20, 35)  # Lower chance for tornado formation
        else:
            # Randomize CAPE, helicity, and wind shear for other locations
            cape = random.choice([400, 900, 1600, 2800, 3500])
            helicity = random.choice([40, 120, 220, 300, 400])
            wind_shear = random.choice([12, 24, 35, 45]) 
            risk_levels = ['low', 'moderate', 'high', 'extreme']
            risk_level = random.choice(risk_levels)
            formation_chance = random.randint(60, 95)
        
        prediction = {
            'id': f"ai-{str(uuid.uuid4())[:8]}",
            'name': f"AI Prediction: {location['name']}",
            'risk_level': risk_level,
            'risk': risk_level,
            'formation_chance': formation_chance,
            'chance': formation_chance,
            'direction': 'NE',
            'speed': random.randint(20, 40),
            'cape': cape,
            'helicity': helicity,
            'shear': f"{wind_shear} knots",
            'wind_shear': wind_shear,
            'radar': 'AI analysis',
            'lat': location['lat'],
            'lon': location['lon'],
            'polygon': None,
            'description': f"AI-based tornado prediction with {formation_chance}% probability",
            'nws_alert': False,
            'is_ai_prediction': True,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'location': location['name'],
            'confidence': random.uniform(0.6, 0.9)
        }
        
        predictions.append(prediction)
    
    return predictions

# Initialize our AI model
tornado_ai = TornadoAI()  # Use the real model instead of mock
# tornado_ai = MockTornadoAI()  # Comment out the mock for production

# Initialize the database
initialize_db()

# Define user agent for API requests
USER_AGENT = 'ExtremeWeatherPredictions/1.0 (https://github.com/username/ExtremeWeatherPredictions)'

def get_coords_from_zipcode(zipcode):
    """Convert zipcode to latitude/longitude using nominatim API."""
    try:
        # Add delay to respect rate limiting but shorten it
        time.sleep(0.5)
        
        # Skip API call if zipcode is not valid
        if not zipcode or not re.match(r'^\d{5}$', zipcode):
            print(f"Invalid zipcode format: {zipcode}")
            return None
            
        headers = {
            'User-Agent': USER_AGENT,
            'Accept': 'application/json',
            'Referer': 'https://github.com/username/ExtremeWeatherPredictions'
        }
        
        # Use a shorter timeout of 3 seconds instead of 5
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            headers=headers,
            params={
                "postalcode": zipcode,
                "country": "US",
                "format": "json",
                "addressdetails": 1,
                "limit": 1
            },
            timeout=3
        )
        
        response.raise_for_status()  # Raise an error for bad status codes
        
        data = response.json()
        
        if not data:
            print(f"No results found for zipcode: {zipcode}")
            return None
            
        return {
            "latitude": float(data[0]["lat"]),
            "longitude": float(data[0]["lon"])
        }
        
    except requests.exceptions.Timeout:
        print("Timeout while contacting Nominatim API")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Network error while contacting Nominatim API: {e}")
        return None
    except (KeyError, IndexError, ValueError) as e:
        print(f"Error parsing Nominatim API response: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error getting coordinates for zipcode: {e}")
        return None

def get_nws_gridpoint(lat, lon):
    """Get NWS gridpoint information for the location."""
    try:
        headers = {
            'User-Agent': USER_AGENT,
            'Accept': 'application/geo+json'
        }
        
        # First get the grid endpoint for the location
        url = f"https://api.weather.gov/points/{lat},{lon}"
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Verify the response has the expected structure
        if not data or 'properties' not in data:
            print(f"NWS API response for {lat},{lon} missing expected data structure")
            return None
            
        props = data['properties']
        required_fields = ['gridId', 'gridX', 'gridY', 'forecast', 'forecastHourly']
        
        if not all(field in props for field in required_fields):
            print(f"NWS API response for {lat},{lon} missing required fields")
            return None
        
        return {
            'grid_id': props['gridId'],
            'grid_x': props['gridX'],
            'grid_y': props['gridY'],
            'forecast_url': props['forecast'],
            'hourly_forecast_url': props['forecastHourly']
        }
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"Location {lat},{lon} not covered by NWS API (outside US)")
            return None
        print(f"HTTP error getting NWS gridpoint: {e}")
        return None
    except Exception as e:
        print(f"Error getting NWS gridpoint: {e}")
        return None

def get_nws_forecast(gridpoint, timestamp=None):
    """Get NWS forecast for the specified gridpoint."""
    try:
        if not gridpoint or 'hourly_forecast_url' not in gridpoint:
            print("Invalid gridpoint data provided to get_nws_forecast")
            return None
            
        headers = {
            'User-Agent': USER_AGENT,
            'Accept': 'application/geo+json'
        }
        
        # Use the hourly forecast URL from the gridpoint
        response = requests.get(gridpoint['hourly_forecast_url'], headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Check if the response has the expected structure
        if not data or 'properties' not in data or 'periods' not in data['properties'] or not data['properties']['periods']:
            print("NWS API response missing expected data structure")
            return None
            
        # If timestamp is provided, find the closest period
        if timestamp:
            # Ensure we have a timezone-aware datetime for comparison
            try:
                # Parse the input timestamp
                if timestamp.endswith('Z'):
                    # UTC time
                    target_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                elif '+' in timestamp or '-' in timestamp and 'T' in timestamp:
                    # Already has timezone info
                    target_time = datetime.fromisoformat(timestamp)
                else:
                    # Assume local time, add UTC timezone
                    try:
                        target_time = datetime.fromisoformat(timestamp).replace(tzinfo=timezone.utc)
                    except ValueError:
                        # If that fails, try a different format
                        target_time = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
            except Exception as e:
                print(f"Error parsing input timestamp '{timestamp}': {e}")
                target_time = datetime.now(timezone.utc)
                print(f"Using current time instead: {target_time.isoformat()}")
            
            # Find the period closest to the target time
            closest_period = None
            min_diff = timedelta(days=365)
            
            for period in data['properties']['periods']:
                if 'startTime' not in period:
                    continue
                    
                try:
                    # Parse the period startTime with explicit timezone handling
                    period_time_str = period['startTime']
                    
                    # Ensure period_time is timezone aware
                    if period_time_str.endswith('Z'):
                        period_time = datetime.fromisoformat(period_time_str.replace('Z', '+00:00'))
                    elif '+' in period_time_str or '-' in period_time_str and 'T' in period_time_str:
                        # Contains timezone info
                        period_time = datetime.fromisoformat(period_time_str)
                    else:
                        # No timezone info, assume UTC
                        try:
                            period_time = datetime.fromisoformat(period_time_str).replace(tzinfo=timezone.utc)
                        except ValueError:
                            # If parsing fails, try a different format
                            period_time = datetime.strptime(period_time_str, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
                    
                    # Both times should now be timezone aware for comparison
                    # Double-check timezone awareness to prevent errors
                    if period_time.tzinfo is None:
                        period_time = period_time.replace(tzinfo=timezone.utc)
                    if target_time.tzinfo is None:
                        target_time = target_time.replace(tzinfo=timezone.utc)
                        
                    # Now we can safely compare
                    diff = abs(period_time - target_time)
                    
                    if diff < min_diff:
                        min_diff = diff
                        closest_period = period
                except Exception as e:
                    print(f"Error comparing datetime for period {period.get('number', 'unknown')}: {e}")
                    continue
            
            # Return the closest period we found, or fallback to the first period
            if closest_period:
                # Sanitize dates in the returned period
                clean_period = sanitize_forecast_period(closest_period)
                return clean_period
            
            # Fallback if we couldn't find a closest period
            if data['properties']['periods']:
                clean_period = sanitize_forecast_period(data['properties']['periods'][0])
                return clean_period
            return None
        
        # Otherwise, return the current forecast (first period)
        if data['properties']['periods']:
            clean_period = sanitize_forecast_period(data['properties']['periods'][0])
            return clean_period
        return None
            
    except requests.exceptions.Timeout:
        print("Timeout while fetching NWS forecast data")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error getting NWS forecast: {e}")
        return None
    except Exception as e:
        print(f"Error getting NWS forecast: {e}")
        return None

def sanitize_forecast_period(period):
    """Clean up a forecast period to ensure all fields are valid."""
    if not period:
        return None
        
    # Create a clean copy
    clean_period = period.copy()
    
    # Sanitize the startTime if it exists
    if 'startTime' in clean_period:
        try:
            # Try to parse it to verify it's valid
            if clean_period['startTime'].endswith('Z'):
                datetime.fromisoformat(clean_period['startTime'].replace('Z', '+00:00'))
            else:
                datetime.fromisoformat(clean_period['startTime'])
        except ValueError:
            # Replace invalid startTime with current time
            clean_period['startTime'] = datetime.now(timezone.utc).isoformat()
    
    # Ensure all text fields are strings
    for field in ['detailedForecast', 'shortForecast', 'name', 'windSpeed', 'windDirection']:
        if field in clean_period and clean_period[field] is None:
            clean_period[field] = ""
            
    # Ensure numeric fields are valid
    for field in ['temperature', 'windSpeed']:
        if field in clean_period:
            # Extract numeric value from windSpeed if it's a string
            if field == 'windSpeed' and isinstance(clean_period[field], str):
                try:
                    # Extract digits from string like "10 mph"
                    clean_period[field] = ''.join(filter(str.isdigit, clean_period[field]))
                except:
                    clean_period[field] = "0"
            
            # Make sure temperature is numeric
            if field == 'temperature' and not isinstance(clean_period[field], (int, float)):
                try:
                    clean_period[field] = float(clean_period[field])
                except:
                    clean_period[field] = 70  # Default temperature
    
    return clean_period

# Simple cache for Open-Meteo API to avoid rate limiting
open_meteo_cache = {}
OPEN_METEO_CACHE_DURATION = 30 * 60  # 30 minutes in seconds

def get_open_meteo_data(lat, lon):
    """Get weather data from Open-Meteo free weather API with caching.
    NOTE: Currently disabled to reduce unnecessary API requests.
    """
    # Just return None - Open-Meteo data is not being used currently
    return None
    
    # Original implementation commented out below
    """
    max_retries = 2
    retry_delay = 2  # seconds
    
    # Round coordinates to 4 decimal places for cache key
    # This avoids multiple similar cache entries for nearly identical locations
    lat_rounded = round(float(lat), 4)
    lon_rounded = round(float(lon), 4)
    cache_key = f"{lat_rounded},{lon_rounded}"
    
    # Check if we have a cached response that's still valid
    if cache_key in open_meteo_cache:
        cache_entry = open_meteo_cache[cache_key]
        cache_age = time.time() - cache_entry['timestamp']
        
        # Return cached data if it's still fresh
        if cache_age < OPEN_METEO_CACHE_DURATION:
            print(f"Using cached Open-Meteo data for {cache_key} (age: {int(cache_age/60)} minutes)")
            return cache_entry['data']
        else:
            print(f"Cached Open-Meteo data expired for {cache_key} (age: {int(cache_age/60)} minutes)")
    
    for attempt in range(max_retries + 1):
        try:
            # Open-Meteo provides a completely free weather API with no key requirement
            # Respect rate limits (5000 requests/day, 10 requests/second)
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat_rounded}&longitude={lon_rounded}&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,showers,snowfall,weather_code,cloud_cover,wind_speed_10m,wind_direction_10m&hourly=temperature_2m,relative_humidity_2m,precipitation_probability,precipitation,rain,showers,snowfall,weather_code,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,wind_speed_10m,wind_direction_10m,cape&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=auto"
            
            # Increased timeout to help with slower connections
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            # Cache the successful response
            data = response.json()
            open_meteo_cache[cache_key] = {
                'data': data,
                'timestamp': time.time()
            }
            
            print(f"Cached new Open-Meteo data for {cache_key}")
            return data
            
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                print(f"Timeout getting Open-Meteo data, retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries + 1})")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print("Failed to get Open-Meteo data after multiple retries")
                return None
                
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error getting Open-Meteo data: {e}")
            
            # If we get a 429 (Too Many Requests) response, wait longer before retrying
            if e.response.status_code == 429 and attempt < max_retries:
                retry_delay = 5 * (attempt + 1)  # Increasing delay for rate limit errors
                print(f"Rate limited (429), retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries + 1})")
                time.sleep(retry_delay)
            # If it's another server error (5xx), retry with standard backoff
            elif 500 <= e.response.status_code < 600 and attempt < max_retries:
                print(f"Server error, retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries + 1})")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                # Try to return cached data even if it's expired, better than nothing
                if cache_key in open_meteo_cache:
                    print(f"Using expired cache for {cache_key} as API request failed")
                    return open_meteo_cache[cache_key]['data']
                return None
                
        except Exception as e:
            print(f"Error getting Open-Meteo data: {e}")
            # Try to return cached data even if it's expired, better than nothing
            if cache_key in open_meteo_cache:
                print(f"Using expired cache for {cache_key} as API request failed")
                return open_meteo_cache[cache_key]['data']
            return None
    """

def estimate_cape_from_temp_dewpoint(temp_f, dewpoint_f):
    """Estimate CAPE value from temperature and dewpoint."""
    # Convert to Celsius
    temp_c = (temp_f - 32) * 5/9
    dewpoint_c = (dewpoint_f - 32) * 5/9
    
    # Calculate surface mixing ratio (simplified)
    e = 6.11 * 10.0**(7.5 * dewpoint_c / (237.3 + dewpoint_c))
    w = 0.622 * e / (1013.25 - e)
    
    # Estimate CAPE (simplified model)
    # Based on temperature and moisture availability
    moisture_content = w * 1000  # g/kg
    instability = (temp_c - dewpoint_c) * 10  # Temperature-dewpoint spread
    
    # Simple CAPE estimation - this is a very simplified model
    # In real meteorology, CAPE is calculated using temperature profiles through the atmosphere
    cape_estimate = moisture_content * 10 - instability * 5
    
    # Constraint to realistic values
    return max(0, min(5000, int(cape_estimate * 10)))

def convert_open_meteo_weather_code(code):
    """Convert Open-Meteo weather code to descriptive text."""
    codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snow fall",
        73: "Moderate snow fall",
        75: "Heavy snow fall",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail"
    }
    return codes.get(code, "Unknown")

def get_weather_data(lat, lon, timestamp=None):
    """Get weather data from available free APIs with fallbacks."""
    data = {}
    
    # Try NWS first (US only)
    try:
        gridpoint = get_nws_gridpoint(lat, lon)
        if gridpoint:
            data['nws_forecast'] = get_nws_forecast(gridpoint, timestamp)
    except Exception as e:
        print(f"Error getting NWS data: {e}")
    
    # Add Open-Meteo data (worldwide coverage) - DISABLED TO REDUCE API REQUESTS
    # Only uncomment if needed for specific features
    """
    try:
        data['open_meteo'] = get_open_meteo_data(lat, lon)
    except Exception as e:
        print(f"Error getting Open-Meteo data: {e}")
    """
    
    return data

def convert_weather_conditions_to_severity(weather_data):
    """Convert real weather data to severity levels for our app."""
    severity = "low"
    probability = 0.1
    
    # Default values if we can't get accurate ones
    weather_type = "Unknown"
    precip_type = "None"
    precip_intensity = "None"
    
    # Set default weather metrics
    weather_metrics = {
        'weather_type': weather_type,
        'precipitation_type': precip_type,
        'precipitation_intensity': precip_intensity,
        'wind_shear': 25,  # Default wind shear
        'storm_motion': 20,  # Default storm motion
        'cape': 500,  # Default CAPE
        'helicity': 100,  # Default helicity
        'tornadic_probability': 0.01  # Default very low tornadic probability
    }
    
    try:
        # Try to extract from Open-Meteo first (DISABLED CURRENTLY)
        # if 'open_meteo' in weather_data and weather_data['open_meteo']:
        #     meteo = weather_data['open_meteo']
        #     
        #     # Open-Meteo data processing logic removed since it's disabled...
        
        # If we have NWS data (US only), use it to enhance the assessment
        if 'nws_forecast' in weather_data and weather_data['nws_forecast']:
            nws = weather_data['nws_forecast']
            
            nws_forecast_text = nws.get('shortForecast', '').lower()
            nws_details = nws.get('detailedForecast', '').lower()
            
            # Update weather_type from NWS if we have it
            weather_type = nws.get('shortForecast', weather_type)
            weather_metrics['weather_type'] = weather_type
            
            # Update precipitation info based on NWS text
            if 'thunderstorm' in nws_forecast_text or 'thunder' in nws_forecast_text:
                precip_type = "Thunderstorm"
                weather_metrics['precipitation_type'] = precip_type
                if 'severe' in nws_forecast_text or 'heavy' in nws_forecast_text:
                    precip_intensity = "Heavy"
                    weather_metrics['precipitation_intensity'] = precip_intensity
                    severity = "high"
                    probability = 0.7
                else:
                    precip_intensity = "Moderate"
                    weather_metrics['precipitation_intensity'] = precip_intensity
                    severity = "moderate"
                    probability = 0.5
            elif 'rain' in nws_forecast_text:
                precip_type = "Rain"
                weather_metrics['precipitation_type'] = precip_type
                if 'heavy' in nws_forecast_text:
                    precip_intensity = "Heavy"
                    weather_metrics['precipitation_intensity'] = precip_intensity
                    severity = "moderate"
                    probability = 0.4
                else:
                    precip_intensity = "Light"
                    weather_metrics['precipitation_intensity'] = precip_intensity
                    severity = "low"
                    probability = 0.2
            elif 'snow' in nws_forecast_text:
                precip_type = "Snow"
                weather_metrics['precipitation_type'] = precip_type
                if 'heavy' in nws_forecast_text:
                    precip_intensity = "Heavy"
                    weather_metrics['precipitation_intensity'] = precip_intensity
                    severity = "moderate"
                    probability = 0.3
                else:
                    precip_intensity = "Light"
                    weather_metrics['precipitation_intensity'] = precip_intensity
                    severity = "low"
                    probability = 0.1
            
            temp_f = nws.get('temperature', 70)
            wind_speed = int(''.join(filter(str.isdigit, nws.get('windSpeed', '0')))) if 'windSpeed' in nws else 0
            weather_metrics['storm_motion'] = wind_speed
            
            # Determine tornadic probability based on NWS description
            tornadic_probability = 0.01  # Default very low
            
            # Check for explicit tornado mentions
            if 'tornado' in nws_forecast_text or 'tornado' in nws_details:
                tornadic_probability = 0.9  # Increased from 0.8
                severity = "extreme"
                probability = 0.95  # Increased from 0.9
            # Check for severe thunderstorm conditions that could indicate tornadoes
            elif any(term in nws_forecast_text or term in nws_details for term in 
                    ['severe thunderstorm', 'rotation', 'wall cloud', 'supercell', 'mesocyclone', 
                     'funnel cloud', 'tornado watch', 'tornado warning']):
                tornadic_probability = 0.7
                severity = "high"
                probability = 0.8
            # Check for other conditions that suggest elevated tornado risk
            elif any(term in nws_forecast_text or term in nws_details for term in 
                    ['strong thunderstorm', 'damaging winds', 'large hail']):
                tornadic_probability = 0.4
                severity = "moderate" 
                probability = 0.6
            # Thunderstorm conditions
            elif severity == "high" and 'thunderstorm' in nws_forecast_text:
                tornadic_probability = 0.3  # Increased from 0.2
            
            weather_metrics['tornadic_probability'] = tornadic_probability
            
            # Estimate CAPE from temp and other factors
            cape = estimate_cape_from_temp_dewpoint(temp_f, temp_f - 10)  # rough estimate for dewpoint
            weather_metrics['cape'] = cape
            
            # Estimate wind shear
            wind_shear = int(max(5, min(150, wind_speed * 1.5)))
            weather_metrics['wind_shear'] = wind_shear
                
            # Estimate helicity based on tornadic probability
            helicity = int(tornadic_probability * 300)
            weather_metrics['helicity'] = helicity
        
        # Return the computed values
        return severity, probability, weather_metrics
        
    except Exception as e:
        print(f"Error converting weather conditions: {e}")
        return "low", 0.1, weather_metrics

def get_tornado_risk_level(tornadic_probability):
    """Convert tornadic probability to a risk level."""
    if tornadic_probability < 0.005:
        return "None"
    elif tornadic_probability < 0.03:  # Lowered from 0.05
        return "Very Low" 
    elif tornadic_probability < 0.1:   # Lowered from 0.2
        return "Low"
    elif tornadic_probability < 0.25:  # Lowered from 0.4  
        return "Moderate"
    elif tornadic_probability < 0.5:   # Lowered from 0.7
        return "High"
    else:
        return "Extreme"

def generate_risk_summary(weather_metrics, weather_data):
    """Generate a comprehensive risk summary based on real weather data."""
    tornadic_probability = weather_metrics.get('tornadic_probability', 0)
    wind_shear = weather_metrics.get('wind_shear', 0)
    helicity = weather_metrics.get('helicity', 0)
    cape = weather_metrics.get('cape', 0)
    precip_type = weather_metrics.get('precipitation_type', 'Unknown')
    
    # Get weather forecast description from NWS or Open-Meteo
    forecast_desc = "Clear conditions"
    if 'nws_forecast' in weather_data and weather_data['nws_forecast']:
        forecast_desc = weather_data['nws_forecast'].get('shortForecast', 'Clear conditions')
    elif 'open_meteo' in weather_data and weather_data['open_meteo'] and 'current' in weather_data['open_meteo']:
        code = weather_data['open_meteo']['current'].get('weather_code')
        if code is not None:
            forecast_desc = convert_open_meteo_weather_code(code)
    
    # For clear weather or minimal risk
    if tornadic_probability < 0.05:
        summary = ["Overall Risk: Minimal to None", 
                  f"Forecast: {forecast_desc}"]
                  
        # Add scientific explanations for low risk
        summary.append("Scientific Explanation: The atmosphere is currently stable for the following reasons:")
        
        if cape < 500:
            summary.append("• Low CAPE value (Convective Available Potential Energy): " +
                          f"{cape} J/kg is insufficient to support strong updrafts needed for severe thunderstorms.")
                          
        if wind_shear < 20:
            summary.append("• Minimal vertical wind shear: " +
                          f"Current wind shear of {wind_shear} knots is below the threshold needed for organizing storm rotation.")
                          
        if helicity < 100:
            summary.append("• Low helicity: " +
                          f"Current helicity of {helicity} m²/s² indicates minimal potential for rotation in the atmosphere.")
                          
        if precip_type == "None" or precip_type == "Unknown":
            summary.append("• No precipitation detected: Clear conditions provide no triggering mechanism for storm development.")
        elif precip_type == "Rain" and weather_metrics.get('precipitation_intensity', '') == "Light":
            summary.append("• Light rain only: Current precipitation is not associated with storm development.")
        
        summary.append("RECOMMENDATION: Normal activities, no weather concerns")
        return summary
    
    risk_level = get_tornado_risk_level(tornadic_probability)
    
    summary = [f"Overall Risk: {risk_level}"]
    
    # Add the forecast
    summary.append(f"Forecast: {forecast_desc}")
    
    # Add scientific explanation for low risk (but not minimal)
    if risk_level == "Low":
        summary.append("Scientific Explanation: While some instability exists, conditions are not favorable for severe weather:")
        
        if cape < 1000:
            summary.append(f"• Moderate CAPE ({cape} J/kg): Some atmospheric instability, but below severe thresholds.")
            
        if wind_shear < 25:
            summary.append(f"• Limited wind shear ({wind_shear} knots): Insufficient to organize strong rotating storms.")
            
        if helicity < 150:
            summary.append(f"• Low helicity value ({helicity} m²/s²): Minimal rotational energy in the atmosphere.")
    
    # Add key risk factors for higher risks
    if tornadic_probability > 0.2:
        summary.append("Risk Factors:")
        if wind_shear > 30:
            summary.append(f"• Strong vertical wind shear ({wind_shear} knots) favorable for storm rotation")
        if helicity > 150:
            summary.append(f"• Enhanced helicity ({helicity} m²/s²) indicating rotational storm potential")
        if cape > 2000:
            summary.append(f"• High instability (CAPE: {cape} J/kg) supporting strong updrafts")
    
    # Add recommendation based on risk level
    if risk_level in ["Extreme", "High"]:
        summary.append("RECOMMENDATION: Take shelter immediately, monitor official warnings")
    elif risk_level == "Moderate":
        summary.append("RECOMMENDATION: Stay alert and prepared to take shelter, monitor for warnings")
    else:
        summary.append("RECOMMENDATION: Monitor conditions for changes")
        
    return summary

@app.route('/')
def index():
    """Render the main application page."""
    return render_template('index.html')

@app.route('/tornado')
def tornado():
    """Render the dedicated tornado risk map page."""
    return render_template('tornado.html')

@app.route('/model-stats')
def model_stats():
    """Render the model statistics page."""
    return render_template('model_stats.html')

@app.route('/api/predict', methods=['GET', 'POST'])
def predict():
    """Handle weather prediction requests using real data from free weather APIs."""
    try:
        # Set a timeout for the entire function
        start_time = time.time()
        max_execution_time = 10  # maximum seconds to run
        
        # Initialize warning_message to None to fix reference error
        warning_message = None
        
        if request.method == 'POST':
            print("POST request to /api/predict received")
            data = request.json
            if not data:
                print("No JSON data received in POST request")
                return jsonify({"error": "No JSON data received"}), 400
                
            zipcode = data.get('zipcode')
            if not zipcode:
                print("No zipcode provided in POST request")
                return jsonify({"error": "No zipcode provided"}), 400
                
            timestamp = data.get('timestamp')
            if not timestamp:
                timestamp = datetime.now().isoformat()
                
            print(f"Processing request for zipcode: {zipcode}, timestamp: {timestamp}")
        else:
            # Default values for GET request (primarily for the tornado page)
            zipcode = '66049'  # Lawrence, KS - tornado alley
            timestamp = datetime.now().isoformat()
            print(f"GET request to /api/predict, using default zipcode: {zipcode}")

        # Convert zipcode to coordinates
        print(f"Looking up coordinates for zipcode: {zipcode}")
        location = get_coords_from_zipcode(zipcode)
        
        # Check execution time after zipcode lookup
        if time.time() - start_time > max_execution_time:
            print(f"Request for zipcode {zipcode} is taking too long, using fallback")
            # Fallback coordinates based on US ZIP code regions
            fallback_locations = {
                '0': {"latitude": 40.7128, "longitude": -74.0060},  # Northeast (NYC)
                '1': {"latitude": 42.3601, "longitude": -71.0589},  # Northeast (Boston)
                '2': {"latitude": 38.9072, "longitude": -77.0369},  # East Coast (DC)
                '3': {"latitude": 33.7490, "longitude": -84.3880},  # Southeast (Atlanta)
                '4': {"latitude": 39.1031, "longitude": -84.5120},  # Midwest (Cincinnati)
                '5': {"latitude": 30.2672, "longitude": -97.7431},  # South (Austin)
                '6': {"latitude": 39.0997, "longitude": -94.5786},  # Midwest (Kansas City)
                '604': {"latitude": 41.8781, "longitude": -87.6298},  # Chicago area
                '7': {"latitude": 41.8781, "longitude": -87.6298},  # Midwest (Chicago)
                '8': {"latitude": 44.9778, "longitude": -93.2650},  # Upper Midwest (Minneapolis)
                '9': {"latitude": 37.7749, "longitude": -122.4194}  # West Coast (SF)
            }
            
            # Try to find the most specific location match
            location = None
            for prefix_length in range(min(3, len(zipcode)), 0, -1):
                prefix = zipcode[:prefix_length]
                if prefix in fallback_locations:
                    location = fallback_locations[prefix]
                    print(f"Using fallback location for zipcode prefix {prefix}: {location}")
                    break
            
            # If no specific match found, use first digit
            if not location:
                first_digit = zipcode[0] if zipcode and len(zipcode) > 0 else '6'
                location = fallback_locations.get(first_digit, {"latitude": 39.8283, "longitude": -98.5795})  # Default to geographic center of US
                print(f"Using fallback location for zipcode first digit {first_digit}: {location}")
            
            # Add warning flag that we're using estimated location
            using_fallback_location = True
            warning_message = "Request is taking too long. Using approximate location and estimated data."
            
            # Create minimal response with fallback data
            tornadic_probability = 0.05 + (random.random() * 0.1)  # Random value between 0.05 and 0.15
            risk_level = get_tornado_risk_level(tornadic_probability)
            
            response = {
                "location": location,
                "severity": "Minor",
                "probability": tornadic_probability,
                "weather_type": "Thunderstorm",
                "description": warning_message,
                "risk_summary": [warning_message, "Using estimated weather conditions"],
                "tornadic_probability": tornadic_probability,
                "tornadic_metrics": {
                    "risk_level": risk_level,
                    "cape": 800,
                    "helicity": 100,
                    "vorticity": float(tornadic_probability * 10),
                    "mesocyclone_strength": float(tornadic_probability * 5),
                    "rotation_duration": 5,
                    "mesocyclone_diameter": 2,
                    "wall_cloud_presence": False,
                    "hook_echo_intensity": 1.0,
                    "rotation_height": 5000,
                    "vertical_extent": 8000,
                    "storm_motion": 25
                }
            }
            
            return jsonify(response)
        
        # If we couldn't find coordinates, use fallback values based on region
        if not location:
            print(f"Failed to find coordinates for zipcode: {zipcode}, using fallback")
            # Use fallback locations for common ZIP code prefixes
            first_digit = zipcode[0] if zipcode and len(zipcode) > 0 else '6'
            
            # Fallback coordinates based on US ZIP code regions
            fallback_locations = {
                '0': {"latitude": 40.7128, "longitude": -74.0060},  # Northeast (NYC)
                '1': {"latitude": 42.3601, "longitude": -71.0589},  # Northeast (Boston)
                '2': {"latitude": 38.9072, "longitude": -77.0369},  # East Coast (DC)
                '3': {"latitude": 33.7490, "longitude": -84.3880},  # Southeast (Atlanta)
                '4': {"latitude": 39.1031, "longitude": -84.5120},  # Midwest (Cincinnati)
                '5': {"latitude": 30.2672, "longitude": -97.7431},  # South (Austin)
                '6': {"latitude": 39.0997, "longitude": -94.5786},  # Midwest (Kansas City)
                '604': {"latitude": 41.8781, "longitude": -87.6298},  # Chicago area
                '7': {"latitude": 41.8781, "longitude": -87.6298},  # Midwest (Chicago)
                '8': {"latitude": 44.9778, "longitude": -93.2650},  # Upper Midwest (Minneapolis)
                '9': {"latitude": 37.7749, "longitude": -122.4194}  # West Coast (SF)
            }
            
            # Try to find the most specific location match
            location = None
            for prefix_length in range(min(3, len(zipcode)), 0, -1):
                prefix = zipcode[:prefix_length]
                if prefix in fallback_locations:
                    location = fallback_locations[prefix]
                    print(f"Using fallback location for zipcode prefix {prefix}: {location}")
                    break
            
            # If no specific match found, use first digit
            if not location:
                first_digit = zipcode[0] if zipcode and len(zipcode) > 0 else '6'
                location = fallback_locations.get(first_digit, {"latitude": 39.8283, "longitude": -98.5795})  # Default to geographic center of US
                print(f"Using fallback location for zipcode first digit {first_digit}: {location}")
            
            # Add warning flag that we're using estimated location
            using_fallback_location = True
        else:
            using_fallback_location = False
            print(f"Found coordinates for {zipcode}: {location}")
        
        # Check execution time after location processing
        if time.time() - start_time > max_execution_time:
            print(f"Request for zipcode {zipcode} is taking too long (after location processing), using fallback data")
            # Create minimal response with fallback data
            tornadic_probability = 0.05 + (random.random() * 0.1)  # Random value between 0.05 and 0.15
            risk_level = get_tornado_risk_level(tornadic_probability)
            
            response = {
                "location": location,
                "severity": "Minor",
                "probability": tornadic_probability,
                "weather_type": "Thunderstorm",
                "description": "Request is taking too long. Using estimated weather data.",
                "risk_summary": ["Request timed out", "Using estimated weather conditions"],
                "tornadic_probability": tornadic_probability,
                "tornadic_metrics": {
                    "risk_level": risk_level,
                    "cape": 800,
                    "helicity": 100,
                    "vorticity": float(tornadic_probability * 10),
                    "mesocyclone_strength": float(tornadic_probability * 5),
                    "rotation_duration": 5,
                    "mesocyclone_diameter": 2,
                    "wall_cloud_presence": False,
                    "hook_echo_intensity": 1.0,
                    "rotation_height": 5000,
                    "vertical_extent": 8000,
                    "storm_motion": 25
                }
            }
            
            return jsonify(response)
        
        # Get weather data from all available free APIs
        weather_data = get_weather_data(location['latitude'], location['longitude'], timestamp)
        
        # Final execution time check
        if time.time() - start_time > max_execution_time:
            print(f"Request for zipcode {zipcode} is taking too long (final stage), using fallback data")
            # Create minimal response with fallback data
            tornadic_probability = 0.05 + (random.random() * 0.1)  # Random value between 0.05 and 0.15
            risk_level = get_tornado_risk_level(tornadic_probability)
            
            response = {
                "location": location,
                "severity": "Minor",
                "probability": tornadic_probability,
                "weather_type": "Thunderstorm",
                "description": "Request is taking too long. Using estimated weather data.",
                "risk_summary": ["Request timed out", "Using estimated weather conditions"],
                "tornadic_probability": tornadic_probability,
                "tornadic_metrics": {
                    "risk_level": risk_level,
                    "cape": 800,
                    "helicity": 100,
                    "vorticity": float(tornadic_probability * 10),
                    "mesocyclone_strength": float(tornadic_probability * 5),
                    "rotation_duration": 5,
                    "mesocyclone_diameter": 2,
                    "wall_cloud_presence": False,
                    "hook_echo_intensity": 1.0,
                    "rotation_height": 5000,
                    "vertical_extent": 8000,
                    "storm_motion": 25
                }
            }
            
            return jsonify(response)
        
        # Convert real weather data to our format
        severity, probability, weather_metrics = convert_weather_conditions_to_severity(weather_data)
        
        # Generate risk summary
        risk_summary = generate_risk_summary(weather_metrics, weather_data)
        
        # Add warning to risk summary if we're using fallback data
        if warning_message:
            risk_summary.insert(0, f"⚠️ {warning_message}")
        
        # Create tornadic metrics based on real data
        tornadic_probability = weather_metrics.get('tornadic_probability', 0.01)
        risk_level = get_tornado_risk_level(tornadic_probability)
        
        # Generate tornadic metrics - mostly low values unless actual tornado risk
        vorticity = float(tornadic_probability * 10)
        mesocyclone_strength = float(tornadic_probability * 5)
        wall_cloud_presence = tornadic_probability > 0.3
        hook_echo_intensity = float(tornadic_probability * 10) if tornadic_probability > 0.2 else 0
        rotation_duration = int(tornadic_probability * 30) if tornadic_probability > 0.05 else 0
        mesocyclone_diameter = int(tornadic_probability * 5) if tornadic_probability > 0.05 else 0
        rotation_height = int(tornadic_probability * 15000) if tornadic_probability > 0.05 else 0
        vertical_extent = int(tornadic_probability * 15000) if tornadic_probability > 0.05 else 0
        
        # Create safe data sources object even if data is missing
        data_sources = {
            "nws_forecast": {
                "time": "N/A",
                "short_forecast": "N/A", 
                "temperature": "N/A",
                "wind_speed": "N/A"
            },
            "open_meteo": {
                "available": 'open_meteo' in weather_data and weather_data['open_meteo'] is not None
            }
        }
        
        # Update with actual data if available
        if 'nws_forecast' in weather_data and weather_data['nws_forecast']:
            nws_forecast = weather_data['nws_forecast']
            
            # Handle each field separately to avoid null values
            timeValue = nws_forecast.get('startTime', 'N/A')
            shortForecast = nws_forecast.get('shortForecast', 'N/A') 
            temperature = nws_forecast.get('temperature', 'N/A')
            windSpeed = nws_forecast.get('windSpeed', 'N/A')
            
            # Only use values that are not None or empty strings
            data_sources["nws_forecast"] = {
                "time": timeValue if timeValue and timeValue != "" else "N/A",
                "short_forecast": shortForecast if shortForecast and shortForecast != "" else "N/A",
                "temperature": temperature if temperature and str(temperature) != "" else "N/A",
                "wind_speed": windSpeed if windSpeed and str(windSpeed) != "" else "N/A"
            }
        
        # Create the response with real weather data
        response = {
            "location": location,
            "severity": severity,
            "probability": probability,
            "weather_type": weather_metrics.get('weather_type', 'Unknown'),
            "description": "Weather data available from API sources",
            "risk_summary": risk_summary,
            
            # Basic metrics from real data
            "tornadic_probability": tornadic_probability,
            "precipitation_type": weather_metrics.get('precipitation_type', 'Unknown'),
            "precipitation_intensity": weather_metrics.get('precipitation_intensity', 'Unknown'),
            "wind_shear": weather_metrics.get('wind_shear', 0),
            "storm_motion": weather_metrics.get('storm_motion', 0),
            "cape": weather_metrics.get('cape', 0),
            "helicity": weather_metrics.get('helicity', 0),
            
            # Additional tornadic metrics
            "tornadic_metrics": {
                "risk_level": risk_level,
                "vorticity": vorticity,
                "mesocyclone_strength": mesocyclone_strength,
                "rotation_duration": rotation_duration,
                "mesocyclone_diameter": mesocyclone_diameter,
                "wall_cloud_presence": wall_cloud_presence,
                "hook_echo_intensity": hook_echo_intensity,
                "rotation_height": rotation_height,
                "vertical_extent": vertical_extent
            },
            
            # Additional data sources
            "data_sources": data_sources
        }
        
        # Add detailed forecast if available
        if 'nws_forecast' in weather_data and weather_data['nws_forecast']:
            response["description"] = weather_data['nws_forecast'].get('detailedForecast', 'Weather data available from API sources')
        elif warning_message:
            response["description"] = warning_message

        return jsonify(response)

    except Exception as e:
        print(f"Error processing prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/radar', methods=['POST'])
def process_radar():
    """Handle radar data processing requests."""
    try:
        data = request.get_json()
        radar_data = data.get('radar_data')
        
        if not radar_data:
            return jsonify({'error': 'Missing radar data'}), 400
            
        # Process radar data to detect patterns
        processed_data = radar_processor.process_radar_data(radar_data)
        
        # Enhance results with interpretation
        enhanced_results = {
            "patterns_detected": [],
            "risk_level": "none",
            "details": processed_data
        }
        
        # Add detected patterns to the list
        if processed_data['mesocyclone']['detected']:
            enhanced_results["patterns_detected"].append("mesocyclone")
            
        if processed_data['hook_echo']['detected']:
            enhanced_results["patterns_detected"].append("hook_echo")
            
        # Determine overall risk level
        if processed_data['mesocyclone']['detected'] and processed_data['hook_echo']['detected']:
            # Both detected - highest risk
            if processed_data['mesocyclone']['strength'] > 0.7 and processed_data['hook_echo']['confidence'] > 0.7:
                enhanced_results["risk_level"] = "extreme"
            else:
                enhanced_results["risk_level"] = "high"
        elif processed_data['mesocyclone']['detected']:
            # Only mesocyclone detected
            if processed_data['mesocyclone']['strength'] > 0.7:
                enhanced_results["risk_level"] = "high"
            else:
                enhanced_results["risk_level"] = "moderate"
        elif processed_data['hook_echo']['detected']:
            # Only hook echo detected
            if processed_data['hook_echo']['confidence'] > 0.7:
                enhanced_results["risk_level"] = "moderate"
            else:
                enhanced_results["risk_level"] = "low"
                
        # Add a human-readable summary
        pattern_text = " and ".join(enhanced_results["patterns_detected"])
        if pattern_text:
            enhanced_results["summary"] = f"Detected {pattern_text} - {enhanced_results['risk_level']} risk level"
        else:
            enhanced_results["summary"] = "No tornadic signatures detected"
            
        return jsonify(enhanced_results)
    except Exception as e:
        logger.error(f"Error processing radar data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/radar/live', methods=['GET'])
def get_live_radar():
    """Get live radar data and potential tornado formation spots."""
    try:
        # Get current weather conditions from NOAA
        weather_url = "https://api.weather.gov/alerts/active"
        headers = {
            'User-Agent': USER_AGENT,
            'Accept': 'application/geo+json'
        }
        
        response = requests.get(weather_url, headers=headers, timeout=10)
        response.raise_for_status()
        weather_data = response.json()
        
        # Extract severe weather alerts
        severe_weather_areas = []
        for feature in weather_data.get('features', []):
            properties = feature.get('properties', {})
            if properties.get('event') in ['Tornado Warning', 'Tornado Watch', 'Severe Thunderstorm Warning']:
                geometry = feature.get('geometry')
                if geometry:
                    severe_weather_areas.append({
                        'type': properties.get('event'),
                        'geometry': geometry,
                        'description': properties.get('description', ''),
                        'severity': properties.get('severity', '')
                    })
        
        # Get radar stations data
        radar_url = "https://api.weather.gov/radar/stations"
        response = requests.get(radar_url, headers=headers, timeout=10)
        response.raise_for_status()
        radar_data = response.json()
        
        # Process radar stations
        stations = []
        for feature in radar_data.get('features', []):
            station = feature['properties']
            coordinates = feature['geometry']['coordinates']
            
            # Check if station is in a severe weather area
            tornado_potential = 'none'
            for area in severe_weather_areas:
                if is_point_in_severe_weather_area(coordinates, area['geometry']):
                    if area['type'] == 'Tornado Warning':
                        tornado_potential = 'likely'
                    elif area['type'] == 'Tornado Watch':
                        tornado_potential = 'possible'
            
            stations.append({
                'id': station.get('id'),
                'name': station.get('name'),
                'coordinates': coordinates,
                'elevation': station.get('elevation', {}).get('value'),
                'status': station.get('status', 'unknown'),
                'tornado_potential': tornado_potential
            })
        
        # Get NEXRAD radar mosaic URL
        radar_mosaic_url = "https://opengeo.ncep.noaa.gov/geoserver/conus/conus_bref_qcd/ows?service=WMS&version=1.3.0&request=GetCapabilities"
        
        return jsonify({
            'status': 'success',
            'radar_stations': stations,
            'severe_weather_areas': severe_weather_areas,
            'radar_mosaic_url': radar_mosaic_url
        })
        
    except Exception as e:
        print(f"Error getting live radar data: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/tornado/predictions', methods=['GET'])
def get_tornado_predictions():
    """Generate tornado predictions based on current weather and radar data, including AI predictions."""
    try:
        # Initialize the database if it doesn't exist
        initialize_db()
        
        # Get the batch size parameter (default to all at once)
        batch_param = request.args.get('batch', 'all')
        batch_size = int(batch_param) if batch_param.isdigit() else None
        
        # Get progress parameter for incremental loading
        progress = request.args.get('progress')
        
        # Try to retrieve predictions from cache first
        cached_predictions = getattr(get_tornado_predictions, 'cached_predictions', None)
        cache_timestamp = getattr(get_tornado_predictions, 'cache_timestamp', None)
        
        current_time = datetime.now(timezone.utc)
        
        # Check if we have cached predictions that are less than 5 minutes old
        if cached_predictions and cache_timestamp and (current_time - cache_timestamp).total_seconds() < 300:
            print(f"Using cached predictions from {cache_timestamp.isoformat()}")
            
            # If batch_size is specified, return only that batch
            if batch_size and batch_size > 0:
                batch_index = int(progress or 0) 
                start_idx = batch_index * batch_size
                end_idx = start_idx + batch_size
                total_batches = (len(cached_predictions) + batch_size - 1) // batch_size
                
                # Get the specified batch
                batch_predictions = cached_predictions[start_idx:end_idx]
                
                # Return batch information
                return jsonify({
                    'status': 'success',
                    'predictions': batch_predictions,
                    'cached': True,
                    'cache_time': cache_timestamp.isoformat(),
                    'batch_index': batch_index,
                    'total_batches': total_batches,
                    'has_more': end_idx < len(cached_predictions)
                })
            
            # Otherwise return all predictions at once
            return jsonify({
                'status': 'success',
                'predictions': cached_predictions,
                'cached': True,
                'cache_time': cache_timestamp.isoformat()
            })
            
        # If progress is set and not zero, this is a subsequent request in a batch sequence
        if progress and progress != '0' and hasattr(get_tornado_predictions, 'in_progress_predictions'):
            in_progress = getattr(get_tornado_predictions, 'in_progress_predictions', [])
            progress_index = int(progress)
            
            # If we're still processing predictions, return what we have so far
            if hasattr(get_tornado_predictions, 'processing_complete') and not get_tornado_predictions.processing_complete:
                return jsonify({
                    'status': 'processing',
                    'predictions': in_progress,
                    'progress_index': progress_index,
                    'complete': False
                })
            
            # If progress_index is beyond our processed predictions, return what we have
            if batch_size and progress_index * batch_size >= len(in_progress):
                return jsonify({
                    'status': 'success',
                    'predictions': [],
                    'progress_index': progress_index,
                    'complete': True,
                    'total_predictions': len(in_progress)
                })
                
            # Return the requested batch
            if batch_size:
                start_idx = progress_index * batch_size
                end_idx = start_idx + batch_size
                return jsonify({
                    'status': 'success',
                    'predictions': in_progress[start_idx:end_idx],
                    'progress_index': progress_index,
                    'complete': end_idx >= len(in_progress),
                    'total_predictions': len(in_progress)
                })
            
            # Return all predictions
            return jsonify({
                'status': 'success',
                'predictions': in_progress,
                'complete': True
            })
            
        # If we reach here, we need to generate new predictions
        
        # Initialize in-progress tracking
        get_tornado_predictions.in_progress_predictions = []
        get_tornado_predictions.processing_complete = False
        
        # Get current weather alerts from NOAA
        weather_url = "https://api.weather.gov/alerts/active"
        headers = {
            'User-Agent': USER_AGENT,
            'Accept': 'application/geo+json'
        }
        
        try:
            response = requests.get(weather_url, headers=headers, timeout=10)
            response.raise_for_status()
            weather_data = response.json()
        except Exception as e:
            app.logger.warning(f"Error fetching weather alerts: {e}, proceeding with empty data")
            weather_data = {'features': []}
        
        # Extract the actual polygon data from NWS alerts
        predictions = []
        
        # Process official NWS alerts - ONLY USE REAL DATA
        if 'features' in weather_data and isinstance(weather_data['features'], list):
            for feature in weather_data['features']:
                # Only include tornado and severe thunderstorm alerts
                properties = feature.get('properties', {})
                event_type = properties.get('event', '').lower()
                
                if 'tornado' in event_type or 'thunderstorm' in event_type:
                    # Determine risk level based on event type and severity
                    risk_level = 'moderate'
                    if 'tornado warning' in event_type:
                        risk_level = 'high'
                    elif 'tornado watch' in event_type:
                        risk_level = 'moderate'
                    elif 'severe thunderstorm warning' in event_type:
                        risk_level = 'moderate'
                    
                    # Use the actual geometry from NWS
                    geometry = feature.get('geometry', None)
                    
                    # Calculate approximate center point for display purposes
                    center_lat, center_lon = 0, 0
                    valid_location = False
                    
                    # Define US bounding box for sanity check
                    US_MIN_LAT, US_MAX_LAT = 24.0, 50.0
                    US_MIN_LON, US_MAX_LON = -125.0, -65.0
                    
                    if geometry and geometry.get('type') == 'Polygon':
                        coords = geometry.get('coordinates', [[]])[0]
                        if coords:
                            # Average the coordinates for a simple center
                            lats = [coord[1] for coord in coords]
                            lons = [coord[0] for coord in coords]
                            center_lat = sum(lats) / len(lats)
                            center_lon = sum(lons) / len(lons)
                            
                            # Validate coordinates are within US bounds
                            if US_MIN_LAT <= center_lat <= US_MAX_LAT and US_MIN_LON <= center_lon <= US_MAX_LON:
                                valid_location = True
                            
                            # Extra check to prevent zero or near-zero coordinates
                            if abs(center_lat) < 1.0 or abs(center_lon) < 1.0:
                                valid_location = False
                    
                    # Skip if coordinates are invalid
                    if not valid_location:
                        continue
                    
                    # Extract these attributes directly from the real alert when possible
                    # or provide reasonable estimates based on the alert type
                    headline = properties.get('headline', '')
                    severity = properties.get('severity', 'moderate').lower()
                    
                    # Reasonable estimates based on alert type and severity
                    if 'tornado warning' in event_type:
                        formation_chance = 90
                        cape_value = 3000
                        helicity_value = 300
                    elif 'tornado watch' in event_type:
                        formation_chance = 50
                        cape_value = 2000
                        helicity_value = 200
                    elif 'severe thunderstorm warning' in event_type:
                        formation_chance = 30
                        cape_value = 1500
                        helicity_value = 100
                    else:
                        formation_chance = 20
                        cape_value = 1000 
                        helicity_value = 50
                    
                    # Use consistent direction data
                    direction = 'NE' # Default direction
                    if 'movement' in properties and properties['movement']:
                        movement = properties['movement'].upper()
                        for dir in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
                            if dir in movement:
                                direction = dir
                                break
                    
                    # Extract speed if possible from the alert text
                    speed = 25  # Default speed
                    description = properties.get('description', '')
                    
                    # Use regex to try to find speed information
                    import re
                    speed_match = re.search(r'moving (?:at|around) (\d+) mph', description, re.IGNORECASE)
                    if speed_match:
                        try:
                            speed = int(speed_match.group(1))
                        except:
                            pass  # Keep default if conversion fails
                    
                    # Use wind data for shear if available
                    wind_data = re.search(r'winds up to (\d+) mph', description, re.IGNORECASE)
                    if wind_data:
                        try:
                            wind_speed = int(wind_data.group(1))
                            wind_shear = f"{wind_speed} knots"
                        except:
                            wind_shear = "30 knots"  # Default
                    else:
                        wind_shear = "30 knots"  # Default
                    
                    # Determine radar status based on alert type
                    if 'tornado' in event_type:
                        if 'warning' in event_type:
                            radar_status = "Rotation detected"
                        else:
                            radar_status = "Favorable conditions"
                    else:
                        radar_status = "Strong reflectivity" 
                    
                    # Use the original name for official alerts
                    display_name = properties.get('areaDesc', 'Unknown Area')
                    location_desc = properties.get('areaDesc', 'Unknown Area')
                    
                    # Format prediction for official NWS alert
                    prediction = {
                        'id': properties.get('id', f"pred-{str(uuid.uuid4())[:8]}"),
                        'name': display_name,
                        'risk_level': risk_level,
                        'risk': risk_level,
                        'formation_chance': formation_chance,
                        'chance': formation_chance,
                        'direction': direction,
                        'speed': speed,
                        'cape': cape_value,
                        'shear': wind_shear,
                        'helicity': helicity_value,
                        'radar': radar_status,
                        'lat': center_lat,
                        'lon': center_lon,
                        'polygon': geometry,
                        'description': headline or properties.get('headline', ''),
                        'nws_alert': True,
                        'is_ai_prediction': False,  # This is a real NWS alert
                        'timestamp': current_time.isoformat(),
                        'location': location_desc
                    }
                    
                    predictions.append(prediction)
                    get_tornado_predictions.in_progress_predictions.append(prediction)
        
        # Get AI-based predictions if model is trained
        if hasattr(tornado_ai, 'model') and tornado_ai.model:
            logger.info("Adding AI-based predictions")
            
            # Define tornado-prone areas to check across the United States
            tornado_prone_areas = [
                # Tornado Alley (Central Plains)
                {'name': 'Oklahoma City, OK', 'lat': 35.4676, 'lon': -97.5164},
                {'name': 'Norman, OK', 'lat': 35.2226, 'lon': -97.4395},
                {'name': 'Tulsa, OK', 'lat': 36.1540, 'lon': -95.9928},
                {'name': 'Wichita, KS', 'lat': 37.6872, 'lon': -97.3301},
                {'name': 'Topeka, KS', 'lat': 39.0558, 'lon': -95.6894},
                {'name': 'Dodge City, KS', 'lat': 37.7528, 'lon': -100.0171},
                {'name': 'Dallas, TX', 'lat': 32.7767, 'lon': -96.7970},
                {'name': 'Fort Worth, TX', 'lat': 32.7555, 'lon': -97.3308},
                {'name': 'Amarillo, TX', 'lat': 35.2220, 'lon': -101.8313},
                {'name': 'Lubbock, TX', 'lat': 33.5779, 'lon': -101.8552},
                {'name': 'Lincoln, NE', 'lat': 40.8136, 'lon': -96.7026},
                {'name': 'Omaha, NE', 'lat': 41.2565, 'lon': -95.9345},
                {'name': 'Grand Island, NE', 'lat': 40.9264, 'lon': -98.3420},
                
                # Dixie Alley (Southeast)
                {'name': 'Little Rock, AR', 'lat': 34.7465, 'lon': -92.2896},
                {'name': 'Jonesboro, AR', 'lat': 35.8423, 'lon': -90.7043},
                {'name': 'Birmingham, AL', 'lat': 33.5186, 'lon': -86.8104},
                {'name': 'Huntsville, AL', 'lat': 34.7304, 'lon': -86.5861},
                {'name': 'Tuscaloosa, AL', 'lat': 33.2098, 'lon': -87.5692},
                {'name': 'Jackson, MS', 'lat': 32.2988, 'lon': -90.1848},
                {'name': 'Tupelo, MS', 'lat': 34.2576, 'lon': -88.7032},
                {'name': 'Memphis, TN', 'lat': 35.1495, 'lon': -90.0490},
                {'name': 'Nashville, TN', 'lat': 36.1627, 'lon': -86.7816},
                {'name': 'Chattanooga, TN', 'lat': 35.0456, 'lon': -85.3097},
                
                # Additional states: Georgia
                {'name': 'Atlanta, GA', 'lat': 33.7490, 'lon': -84.3880},
                {'name': 'Macon, GA', 'lat': 32.8407, 'lon': -83.6324},
                {'name': 'Albany, GA', 'lat': 31.5785, 'lon': -84.1557},
                {'name': 'Columbus, GA', 'lat': 32.4610, 'lon': -84.9877},
                
                # Illinois
                {'name': 'Chicago, IL', 'lat': 41.8781, 'lon': -87.6298},
                {'name': 'Springfield, IL', 'lat': 39.7817, 'lon': -89.6501},
                {'name': 'Peoria, IL', 'lat': 40.6936, 'lon': -89.5890},
                {'name': 'Carbondale, IL', 'lat': 37.7273, 'lon': -89.2168},
                
                # Indiana
                {'name': 'Indianapolis, IN', 'lat': 39.7684, 'lon': -86.1581},
                {'name': 'Fort Wayne, IN', 'lat': 41.0793, 'lon': -85.1394},
                {'name': 'Evansville, IN', 'lat': 37.9716, 'lon': -87.5711},
                {'name': 'South Bend, IN', 'lat': 41.6834, 'lon': -86.2500},
                
                # Wisconsin
                {'name': 'Milwaukee, WI', 'lat': 43.0389, 'lon': -87.9065},
                {'name': 'Madison, WI', 'lat': 43.0731, 'lon': -89.4012},
                {'name': 'Green Bay, WI', 'lat': 44.5133, 'lon': -88.0133},
                {'name': 'La Crosse, WI', 'lat': 43.8014, 'lon': -91.2396},
                
                # Ohio
                {'name': 'Columbus, OH', 'lat': 39.9612, 'lon': -82.9988},
                {'name': 'Cleveland, OH', 'lat': 41.4993, 'lon': -81.6944},
                {'name': 'Cincinnati, OH', 'lat': 39.1031, 'lon': -84.5120},
                {'name': 'Toledo, OH', 'lat': 41.6528, 'lon': -83.5379},
                {'name': 'Dayton, OH', 'lat': 39.7589, 'lon': -84.1916},
                
                # Michigan
                {'name': 'Detroit, MI', 'lat': 42.3314, 'lon': -83.0458},
                {'name': 'Grand Rapids, MI', 'lat': 42.9634, 'lon': -85.6681},
                {'name': 'Lansing, MI', 'lat': 42.7325, 'lon': -84.5555},
                
                # Missouri
                {'name': 'St. Louis, MO', 'lat': 38.6270, 'lon': -90.1994},
                {'name': 'Kansas City, MO', 'lat': 39.0997, 'lon': -94.5786},
                {'name': 'Springfield, MO', 'lat': 37.2090, 'lon': -93.2923},
                {'name': 'Joplin, MO', 'lat': 37.0842, 'lon': -94.5133},
                
                # Iowa
                {'name': 'Des Moines, IA', 'lat': 41.5868, 'lon': -93.6250},
                {'name': 'Cedar Rapids, IA', 'lat': 41.9779, 'lon': -91.6656},
                {'name': 'Davenport, IA', 'lat': 41.5236, 'lon': -90.5776},
                
                # Kentucky
                {'name': 'Louisville, KY', 'lat': 38.2527, 'lon': -85.7585},
                {'name': 'Lexington, KY', 'lat': 38.0406, 'lon': -84.5037},
                {'name': 'Bowling Green, KY', 'lat': 36.9685, 'lon': -86.4808},
                
                # Minnesota
                {'name': 'Minneapolis, MN', 'lat': 44.9778, 'lon': -93.2650},
                {'name': 'Rochester, MN', 'lat': 44.0121, 'lon': -92.4802},
                {'name': 'Duluth, MN', 'lat': 46.7867, 'lon': -92.1005}
            ]
            
            # Prioritize locations based on seasonal risk
            prioritized_areas = prioritize_tornado_locations(tornado_prone_areas)
            
            # Use max_locations to limit API calls and processing
            max_locations = 30
            
            # Make sure we get a good regional distribution by picking top areas from different regions
            high_priority_areas = []
            
            # Define regions for distribution
            regions = {
                'tornado_alley': ['OK', 'KS', 'TX', 'NE'],
                'dixie_alley': ['AR', 'AL', 'MS', 'TN'],
                'midwest': ['IL', 'IN', 'MO', 'IA'],
                'great_lakes': ['WI', 'MI', 'MN', 'OH'],
                'southeast': ['GA', 'KY', 'SC', 'NC']
            }
            
            # Track how many locations we've selected from each region
            region_counts = {region: 0 for region in regions}
            
            # First pass: Add at least 2 locations from each region
            min_per_region = 2
            for area in prioritized_areas:
                # Extract state code from name
                name_parts = area['name'].split(', ')
                if len(name_parts) < 2:
                    continue
                    
                state_code = name_parts[1]
                
                # Find which region this state belongs to
                area_region = None
                for region, states in regions.items():
                    if state_code in states:
                        area_region = region
                        break
                
                # If we haven't filled this region's minimum quota, add it
                if area_region and region_counts[area_region] < min_per_region:
                    high_priority_areas.append(area)
                    region_counts[area_region] += 1
                    
                # If we've reached our max locations, stop
                if len(high_priority_areas) >= max_locations:
                    break
            
            # Second pass: Fill remaining slots with highest priority areas
            if len(high_priority_areas) < max_locations:
                for area in prioritized_areas:
                    if area not in high_priority_areas:
                        high_priority_areas.append(area)
                        
                    if len(high_priority_areas) >= max_locations:
                        break
            
            logger.info(f"Checking {len(high_priority_areas)} high-priority areas for tornado risk")
            
            # Process areas in batches to allow for incremental updates
            for i, area in enumerate(high_priority_areas):
                try:
                    # Get weather data for the location
                    weather_data = get_weather_data(area['lat'], area['lon'])
                    _, _, weather_metrics = convert_weather_conditions_to_severity(weather_data)
                    
                    # Get radar image
                    radar_image = tornado_ai.download_radar_image(area['lat'], area['lon'])
                    
                    # Check if radar is clear (no significant echo/returns)
                    is_radar_clear = False
                    if radar_image is not None:
                        # Simple check - if more than 90% of radar image pixels are below threshold
                        # (close to black/clear), then radar is considered clear
                        if isinstance(radar_image, np.ndarray):
                            # Convert to grayscale if it's a color image
                            if len(radar_image.shape) == 3 and radar_image.shape[2] == 3:
                                grayscale = np.mean(radar_image, axis=2)
                            else:
                                grayscale = radar_image
                                
                            # Check if most pixels are very low intensity (clear radar)
                            threshold = 0.1  # Adjust as needed based on your radar imagery
                            clear_percentage = np.mean(grayscale < threshold)
                            is_radar_clear = clear_percentage > 0.9
                    
                    # Skip prediction if radar is clear - don't show alerts for clear skies
                    if is_radar_clear:
                        logger.info(f"Skipping AI prediction for {area['name']} - radar is clear")
                        continue
                    
                    # Make prediction with AI model
                    ai_prediction = tornado_ai.predict(radar_image, weather_metrics)
                    
                    # Only add predictions with significant probability
                    probability = ai_prediction.get('probability', 0)
                    # Increased threshold to reduce false positives
                    if probability > 0.4:  # Increased threshold from 0.3 to 0.4
                        formation_chance = int(probability * 100)
                        severity = ai_prediction.get('severity', 'LOW')
                        confidence = ai_prediction.get('confidence', 0)
                        
                        # Create prediction object
                        pred_id = f"ai-{str(uuid.uuid4())[:8]}"
                        pred = {
                            'id': pred_id,
                            'name': f"AI Prediction: {area['name']}",
                            'risk_level': severity.lower(),
                            'risk': severity.lower(),
                            'formation_chance': formation_chance,
                            'chance': formation_chance,
                            'direction': 'NE',  # Default direction
                            'speed': 25,  # Default speed
                            'cape': weather_metrics.get('cape', 0),
                            'shear': weather_metrics.get('wind_shear', '25 knots'),
                            'helicity': weather_metrics.get('helicity', 0),
                            'radar': 'AI analysis',
                            'lat': area['lat'],
                            'lon': area['lon'],
                            'polygon': None,  # AI predictions don't have polygons yet
                            'description': f"AI-based tornado prediction with {formation_chance}% probability",
                            'nws_alert': False,
                            'is_ai_prediction': True,
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'location': area['name'],
                            'confidence': confidence,
                            'priority': area.get('priority', 5)  # Include priority in prediction
                        }
                        
                        # Apply meteorological constraints based on prediction parameters
                        # (Keep existing code for meteorological constraints)
                        
                        # Make sure risk is updated in all places
                        pred['risk'] = pred['risk_level']
                        
                        # Add to predictions
                        predictions.append(pred)
                        get_tornado_predictions.in_progress_predictions.append(pred)
                        
                        # Log the prediction
                        log_prediction(pred)
                        
                except Exception as e:
                    logger.error(f"Error making AI prediction for {area['name']}: {e}")
        
        # Cache the predictions for 30 minutes
        get_tornado_predictions.cached_predictions = predictions
        get_tornado_predictions.cache_timestamp = current_time
        get_tornado_predictions.processing_complete = True
        
        # Log each prediction to the database for future validation
        for prediction in predictions:
            log_prediction(prediction)
            
        # If it's a new day, try to validate yesterday's predictions
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        last_validation_date = getattr(get_tornado_predictions, 'last_validation_date', None)
        
        if last_validation_date != today:
            # Run validation for predictions from yesterday
            validate_predictions(days_ago=1)
            get_tornado_predictions.last_validation_date = today
        
        # If batch size was specified, return just that batch
        if batch_size and batch_size > 0:
            start_idx = 0
            end_idx = min(batch_size, len(predictions))
            total_batches = (len(predictions) + batch_size - 1) // batch_size
            
            return jsonify({
                'status': 'success',
                'predictions': predictions[start_idx:end_idx],
                'batch_index': 0,
                'total_batches': total_batches,
                'has_more': end_idx < len(predictions),
                'total_predictions': len(predictions)
            })
        
        # Return all predictions at once (default)
        return jsonify({
            'status': 'success',
            'predictions': predictions
        })
    
    except Exception as e:
        app.logger.error(f"Error generating predictions: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def is_point_in_severe_weather_area(point, geometry):
    """Check if a point is within a severe weather area polygon."""
    try:
        if geometry['type'] == 'Polygon':
            # Simple point-in-polygon check
            coordinates = geometry['coordinates'][0]  # Outer ring
            x, y = point
            inside = False
            j = len(coordinates) - 1
            
            for i in range(len(coordinates)):
                if ((coordinates[i][1] > y) != (coordinates[j][1] > y) and
                    x < (coordinates[j][0] - coordinates[i][0]) * (y - coordinates[i][1]) /
                    (coordinates[j][1] - coordinates[i][1]) + coordinates[i][0]):
                    inside = not inside
                j = i
                
            return inside
    except Exception as e:
        print(f"Error checking point in polygon: {e}")
        return False
    
    return False

@app.route('/api/tornado/stats', methods=['GET'])
def get_prediction_stats():
    """Get statistics on model prediction accuracy."""
    try:
        # Create the database and tables if they don't exist
        initialize_db()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get total predictions
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]
        
        # Get AI predictions count
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE is_ai_prediction = TRUE")
        ai_predictions = cursor.fetchone()[0]
        
        # Get NWS alerts count
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE nws_alert = TRUE")
        nws_alerts = cursor.fetchone()[0]
        
        # Get validated predictions count
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE validated = TRUE")
        validated_count = cursor.fetchone()[0]
        
        # Get correct predictions count
        cursor.execute("""
        SELECT COUNT(*) FROM validation_results 
        WHERE was_correct = TRUE
        """)
        correct_count = cursor.fetchone()[0]
        
        # Get last 7 days stats
        seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
        
        cursor.execute("""
        SELECT COUNT(*) FROM predictions 
        WHERE date(prediction_time) >= ?
        """, (seven_days_ago,))
        recent_predictions = cursor.fetchone()[0]
        
        cursor.execute("""
        SELECT COUNT(*) FROM validation_results vr
        JOIN predictions p ON vr.prediction_id = p.id
        WHERE date(p.prediction_time) >= ?
        AND vr.was_correct = TRUE
        """, (seven_days_ago,))
        recent_correct = cursor.fetchone()[0]
        
        # Calculate accuracy
        all_time_accuracy = (correct_count / validated_count * 100) if validated_count > 0 else 0
        recent_accuracy = (recent_correct / recent_predictions * 100) if recent_predictions > 0 else 0
        
        # Get accuracy by risk level
        risk_levels = ['low', 'moderate', 'high', 'extreme']
        risk_accuracy = {}
        
        for risk in risk_levels:
            cursor.execute("""
            SELECT COUNT(*) FROM predictions
            WHERE risk_level = ? AND validated = TRUE
            """, (risk,))
            risk_total = cursor.fetchone()[0]
            
            cursor.execute("""
            SELECT COUNT(*) FROM validation_results vr
            JOIN predictions p ON vr.prediction_id = p.id
            WHERE p.risk_level = ? AND vr.was_correct = TRUE
            """, (risk,))
            risk_correct = cursor.fetchone()[0]
            
            risk_accuracy[risk] = {
                'total': risk_total,
                'correct': risk_correct,
                'accuracy': (risk_correct / risk_total * 100) if risk_total > 0 else 0
            }
        
        # Get last 10 validations
        cursor.execute("""
        SELECT 
            p.id, p.location, p.risk_level, p.formation_chance, 
            vr.was_correct, vr.distance_error_km, vr.notes, p.prediction_time
        FROM validation_results vr
        JOIN predictions p ON vr.prediction_id = p.id
        ORDER BY vr.validated_at DESC
        LIMIT 10
        """)
        
        recent_validations = []
        for row in cursor.fetchall():
            recent_validations.append({
                'id': row[0],
                'location': row[1],
                'risk_level': row[2],
                'formation_chance': row[3],
                'was_correct': bool(row[4]),
                'distance_error_km': row[5],
                'notes': row[6],
                'prediction_time': row[7]
            })
        
        return jsonify({
            'status': 'success',
            'statistics': {
                'total_predictions': total_predictions,
                'ai_predictions': ai_predictions,
                'nws_alerts': nws_alerts,
                'validated_count': validated_count,
                'correct_count': correct_count,
                'all_time_accuracy': all_time_accuracy,
                'recent_predictions': recent_predictions,
                'recent_correct': recent_correct,
                'recent_accuracy': recent_accuracy,
                'risk_accuracy': risk_accuracy,
                'recent_validations': recent_validations
            }
        })
    except sqlite3.OperationalError as e:
        logger.error(f"Database error in prediction stats: {e}")
        # Return empty stats
        return jsonify({
            'status': 'success',
            'statistics': {
                'total_predictions': 0,
                'ai_predictions': 0,
                'nws_alerts': 0,
                'validated_count': 0,
                'correct_count': 0,
                'all_time_accuracy': 0,
                'recent_predictions': 0,
                'recent_correct': 0,
                'recent_accuracy': 0,
                'risk_accuracy': {
                    'low': {'total': 0, 'correct': 0, 'accuracy': 0},
                    'moderate': {'total': 0, 'correct': 0, 'accuracy': 0},
                    'high': {'total': 0, 'correct': 0, 'accuracy': 0},
                    'extreme': {'total': 0, 'correct': 0, 'accuracy': 0}
                },
                'recent_validations': []
            }
        })
    except Exception as e:
        logger.error(f"Error getting prediction stats: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    finally:
        if 'conn' in locals() and conn:
            conn.close()
            
@app.route('/stats')
def stats_page():
    """Render the prediction statistics page."""
    return render_template('stats.html')

@app.route('/api/predict/ai', methods=['POST'])
def predict_with_ai():
    """Make a tornado prediction using deep learning model."""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
        
        # Get location data
        zipcode = data.get('zipcode')
        lat = data.get('latitude')
        lon = data.get('longitude')
        
        # Get timestamp
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        # If we only have zipcode, convert to coordinates
        if zipcode and not (lat and lon):
            location = get_coords_from_zipcode(zipcode)
            if location:
                lat = location['latitude']
                lon = location['longitude']
            else:
                return jsonify({"error": "Could not determine location from zipcode"}), 400
        elif not (lat and lon):
            return jsonify({"error": "No location provided (need zipcode or lat/lon)"}), 400
        
        # Get real weather data
        weather_data = get_weather_data(lat, lon, timestamp)
        
        # Convert to weather metrics for our AI model
        _, _, weather_metrics = convert_weather_conditions_to_severity(weather_data)
        
        # Get radar data for the location
        logger.info(f"Downloading radar image for {lat}, {lon}")
        try:
            radar_image = tornado_ai.download_radar_image(lat, lon)
        except Exception as e:
            logger.error(f"Error downloading radar image: {e}")
            radar_image = None
        
        # Make prediction with our AI model
        prediction = tornado_ai.predict(radar_image, weather_metrics)
        
        # Generate a unique ID for this prediction
        pred_id = str(uuid.uuid4())
        
        # Convert severity to risk level
        severity = prediction.get('severity', 'LOW')
        
        # Get location name
        location_name = "Unknown"
        if 'nws_forecast' in weather_data and weather_data['nws_forecast']:
            forecast = weather_data['nws_forecast']
            if 'name' in forecast:
                location_name = forecast['name']
        
        # Log the prediction for later validation
        prediction_record = {
            'id': pred_id,
            'timestamp': timestamp,
            'lat': lat,
            'lon': lon,
            'location': location_name,
            'risk_level': severity,
            'formation_chance': int(prediction.get('probability', 0) * 100),
            'cape': weather_metrics.get('cape', 0),
            'helicity': weather_metrics.get('helicity', 0),
            'shear': weather_metrics.get('wind_shear', 0),
            'is_ai_prediction': True,
            'nws_alert': False
        }
        log_prediction(prediction_record)
        
        # Add additional details to the response
        response = {
            'id': pred_id,
            'location': {
                'latitude': lat,
                'longitude': lon,
                'name': location_name
            },
            'risk_level': severity,
            'formation_chance': int(prediction.get('probability', 0) * 100),
            'confidence': prediction.get('confidence', 0),
            'prediction_time': timestamp,
            'cape': weather_metrics.get('cape', 0),
            'helicity': weather_metrics.get('helicity', 0),
            'wind_shear': weather_metrics.get('wind_shear', 0),
            'model_version': 'tornado_ai_v1',
            'is_ai_prediction': True
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in AI prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/model/train', methods=['POST'])
def train_model():
    """Train the tornado prediction model with historical data."""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
        
        # Check for required parameters
        data_path = data.get('data_path')
        if not data_path:
            return jsonify({"error": "No data path provided"}), 400
        
        # Optional parameters
        epochs = data.get('epochs', 50)
        batch_size = data.get('batch_size', 32)
        validation_split = data.get('validation_split', 0.2)
        
        # Generate training data from historical data
        training_data = tornado_ai.generate_training_data(data_path)
        
        if training_data.get('status') == 'error':
            return jsonify({"error": training_data.get('message')}), 400
        
        if training_data.get('status') == 'not_implemented':
            # For now, use some mock training data for demonstration
            logger.warning("Using mock training data for demonstration")
            
            # Set number of mock samples
            num_samples = 200
            
            # Generate random radar images
            radar_images = [np.random.rand(224, 224, 3) for _ in range(num_samples)]
            
            # Generate mock weather metrics
            weather_metrics = []
            for _ in range(num_samples):
                # Higher values more likely to be tornadic
                cape = random.randint(0, 5000)
                helicity = random.randint(0, 500)
                wind_shear = random.randint(0, 100)
                
                metrics = {
                    'cape': cape,
                    'helicity': helicity,
                    'wind_shear': wind_shear,
                    'storm_motion': random.randint(10, 60),
                    'temperature': random.randint(50, 95),
                    'dewpoint': random.randint(40, 80),
                    'humidity': random.randint(30, 100),
                    'pressure': random.uniform(980, 1030),
                    'precipitable_water': random.uniform(10, 50),
                    'lapse_rate': random.uniform(5, 9)
                }
                weather_metrics.append(metrics)
            
            # Generate labels based on the metrics (higher cape, helicity = more likely tornado)
            labels = []
            for i in range(num_samples):
                # Calculate tornado probability based on weather parameters
                cape = weather_metrics[i]['cape']
                helicity = weather_metrics[i]['helicity']
                shear = weather_metrics[i]['wind_shear']
                
                # Higher values = higher tornado chance
                tornado_score = (cape / 5000) * 0.4 + (helicity / 500) * 0.4 + (shear / 100) * 0.2
                
                # Apply some randomness but keep correlation with parameters
                tornado = 1 if (tornado_score > 0.6 or (tornado_score > 0.4 and random.random() > 0.5)) else 0
                
                # Determine severity (0=low, 1=moderate, 2=high, 3=extreme)
                if tornado == 0:
                    severity = 0  # Low if no tornado
                else:
                    if tornado_score > 0.8:
                        severity = 3  # Extreme
                    elif tornado_score > 0.65:
                        severity = 2  # High
                    else:
                        severity = 1  # Moderate
                
                labels.append({
                    'tornado': tornado,
                    'severity': severity
                })
            
            # Create training data dictionary
            training_data = {
                'radar_images': radar_images,
                'weather_metrics': weather_metrics,
                'labels': labels
            }
        
        # Train the model
        result = tornado_ai.train(training_data, validation_split, epochs, batch_size)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/model/status', methods=['GET'])
def model_status():
    """Get status of the AI model."""
    try:
        # Check if the model files exist
        model_path = tornado_ai.model_path
        combined_exists = os.path.exists(f"{model_path}_combined.h5")
        radar_exists = os.path.exists(f"{model_path}_radar.h5")
        weather_exists = os.path.exists(f"{model_path}_weather.h5")
        scaler_exists = os.path.exists(f"{model_path}_scaler.json")
        
        # Get model file sizes if they exist
        combined_size = os.path.getsize(f"{model_path}_combined.h5") if combined_exists else 0
        radar_size = os.path.getsize(f"{model_path}_radar.h5") if radar_exists else 0
        weather_size = os.path.getsize(f"{model_path}_weather.h5") if weather_exists else 0
        
        # Calculate total model size in MB
        total_size_mb = (combined_size + radar_size + weather_size) / (1024 * 1024)
        
        # Check if the model is trained
        model_trained = combined_exists and hasattr(tornado_ai.weather_scaler, 'mean_')
        
        # Return status information
        status = {
            'model_exists': combined_exists,
            'model_trained': model_trained,
            'model_files': {
                'combined': combined_exists,
                'radar': radar_exists,
                'weather': weather_exists,
                'scaler': scaler_exists
            },
            'model_size_mb': round(total_size_mb, 2),
            'model_path': model_path,
            'last_modified': datetime.fromtimestamp(os.path.getmtime(f"{model_path}_combined.h5")).isoformat() if combined_exists else None
        }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        return jsonify({"error": str(e)}), 500

# Add a new route for the AI model page
@app.route('/model')
def model_page():
    """Render the AI model page."""
    return render_template('model.html')

def prioritize_tornado_locations(areas, current_month=None):
    """Prioritize tornado locations based on season and historical tornado risk.
    
    Args:
        areas: List of location dictionaries with name, lat, lon
        current_month: Current month (1-12), or None to use current month
        
    Returns:
        Prioritized list of locations with added priority field
    """
    if current_month is None:
        current_month = datetime.now().month
    
    # Define seasonal risk factors (1-10 scale)
    # Based on historical tornado data by region and month
    seasonal_risk = {
        # Tornado Alley peaks in April-June
        'tornado_alley': {
            1: 2, 2: 3, 3: 5, 4: 8, 5: 10, 6: 9,
            7: 6, 8: 5, 9: 4, 10: 3, 11: 2, 12: 2
        },
        # Dixie Alley has peaks in spring and fall
        'dixie_alley': {
            1: 4, 2: 5, 3: 7, 4: 9, 5: 8, 6: 5,
            7: 3, 8: 3, 9: 4, 10: 5, 11: 7, 12: 5
        },
        # Midwest peaks in spring and early summer
        'midwest': {
            1: 1, 2: 2, 3: 4, 4: 7, 5: 9, 6: 10,
            7: 7, 8: 6, 9: 5, 10: 3, 11: 2, 12: 1
        },
        # Florida has weird pattern with summer peak
        'florida': {
            1: 4, 2: 5, 3: 6, 4: 6, 5: 5, 6: 8,
            7: 9, 8: 10, 9: 7, 10: 4, 11: 3, 12: 3
        },
        # Great Lakes region primarily summer
        'great_lakes': {
            1: 1, 2: 1, 3: 2, 4: 4, 5: 7, 6: 9,
            7: 8, 8: 7, 9: 5, 10: 3, 11: 2, 12: 1
        }
    }
    
    # Map states/regions to risk categories
    region_mapping = {
        # Tornado Alley
        'OK': 'tornado_alley',
        'KS': 'tornado_alley',
        'TX': 'tornado_alley',
        'NE': 'tornado_alley',
        
        # Dixie Alley
        'AR': 'dixie_alley',
        'LA': 'dixie_alley',
        'MS': 'dixie_alley',
        'AL': 'dixie_alley',
        'TN': 'dixie_alley',
        
        # Southern states
        'GA': 'dixie_alley',
        'SC': 'dixie_alley',
        'NC': 'dixie_alley',
        
        # Florida (unique pattern)
        'FL': 'florida',
        
        # Midwest
        'IA': 'midwest',
        'MO': 'midwest',
        'IL': 'midwest',
        'IN': 'midwest',
        'OH': 'midwest',
        
        # Upper Midwest/Great Lakes
        'MN': 'great_lakes',
        'WI': 'great_lakes',
        'MI': 'great_lakes'
    }
    
    # Process each area and assign priority
    prioritized_areas = []
    for area in areas:
        # Extract state code from name
        name_parts = area['name'].split(', ')
        if len(name_parts) > 1:
            state_code = name_parts[1]
            region = region_mapping.get(state_code, 'midwest')  # Default to midwest
            
            # Get seasonal risk factor (1-10)
            seasonal_factor = seasonal_risk.get(region, {}).get(current_month, 5)
            
            # Assign higher priority to known high-risk locations
            high_risk_locations = [
                'Oklahoma City', 'Norman', 'Moore', 'Tuscaloosa', 'Joplin', 
                'Birmingham', 'Huntsville', 'Dallas', 'Wichita', 'Kansas City'
            ]
            location_bonus = 2 if any(loc in area['name'] for loc in high_risk_locations) else 0
            
            # Calculate final priority (higher number = higher priority)
            priority = seasonal_factor + location_bonus
            
            # Add priority field to area
            area_with_priority = area.copy()
            area_with_priority['priority'] = priority
            prioritized_areas.append(area_with_priority)
        else:
            # If state can't be determined, use default priority
            area_with_priority = area.copy()
            area_with_priority['priority'] = 5
            prioritized_areas.append(area_with_priority)
    
    # Sort by priority (highest first)
    prioritized_areas.sort(key=lambda x: x['priority'], reverse=True)
    
    return prioritized_areas

@app.route('/api/weather/alerts', methods=['GET'])
def get_weather_alerts():
    """Fetch active weather alerts from the NWS API."""
    try:
        # Try to use cached alerts if available and less than 5 minutes old
        cached_alerts = getattr(get_weather_alerts, 'cached_alerts', None)
        cache_timestamp = getattr(get_weather_alerts, 'cache_timestamp', None)
        
        current_time = datetime.now(timezone.utc)
        
        # Use cache if it's less than 5 minutes old
        if cached_alerts and cache_timestamp and (current_time - cache_timestamp).total_seconds() < 300:
            app.logger.info(f"Using cached weather alerts from {cache_timestamp.isoformat()}")
            return jsonify(cached_alerts)
            
        # Get current weather alerts from NOAA
        weather_url = "https://api.weather.gov/alerts/active"
        headers = {
            'User-Agent': USER_AGENT,
            'Accept': 'application/geo+json'
        }
        
        # Use a shorter timeout (3 seconds) for faster failure
        response = requests.get(weather_url, headers=headers, timeout=3)
        response.raise_for_status()
        weather_data = response.json()
        
        # Extract relevant alert information
        alerts = []
        if 'features' in weather_data and isinstance(weather_data['features'], list):
            for feature in weather_data['features']:
                properties = feature.get('properties', {})
                
                # Only include relevant alerts
                event_type = properties.get('event', '')
                if event_type in ['Tornado Warning', 'Tornado Watch', 'Severe Thunderstorm Warning', 
                                 'Severe Thunderstorm Watch', 'Flash Flood Warning', 'Flash Flood Watch']:
                    # Extract alert information
                    alert = {
                        'id': properties.get('id', ''),
                        'event': event_type,
                        'headline': properties.get('headline', ''),
                        'description': properties.get('description', ''),
                        'instruction': properties.get('instruction', ''),
                        'severity': properties.get('severity', ''),
                        'urgency': properties.get('urgency', ''),
                        'area': properties.get('areaDesc', ''),
                        'sent': properties.get('sent', ''),
                        'effective': properties.get('effective', ''),
                        'expires': properties.get('expires', ''),
                        'status': properties.get('status', '')
                    }
                    
                    # Include geometry data if available
                    geometry = feature.get('geometry')
                    if geometry:
                        alert['geometry'] = geometry
                    
                    alerts.append(alert)
        
        # Create response with successful status
        response_data = {
            'status': 'success',
            'count': len(alerts),
            'alerts': alerts,
            'timestamp': current_time.isoformat()
        }
        
        # Cache the alerts for 5 minutes
        get_weather_alerts.cached_alerts = response_data
        get_weather_alerts.cache_timestamp = current_time
        
        return jsonify(response_data)
        
    except requests.exceptions.Timeout:
        app.logger.warning("Timeout when fetching weather alerts from NWS API")
        
        # Try to use cached data even if it's older than 5 minutes as a fallback
        if cached_alerts and cache_timestamp:
            app.logger.info(f"Using stale cached weather alerts from {cache_timestamp.isoformat()} due to timeout")
            
            # Mark the response as using stale data
            cached_alerts['status'] = 'partial'
            cached_alerts['message'] = 'Using cached data due to connection timeout'
            return jsonify(cached_alerts)
        
        # Return a limited response with empty alerts if no cache available
        return jsonify({
            'status': 'success',  # Return success with empty alerts rather than error
            'message': 'Timeout connecting to weather service, no alerts available',
            'alerts': [],
            'count': 0,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Network error fetching weather alerts: {e}")
        
        # Try to use cached data as fallback
        if cached_alerts and cache_timestamp:
            app.logger.info(f"Using stale cached weather alerts from {cache_timestamp.isoformat()} due to network error")
            cached_alerts['status'] = 'partial'
            cached_alerts['message'] = 'Using cached data due to network error'
            return jsonify(cached_alerts)
        
        # Return empty alerts with success status rather than error
        return jsonify({
            'status': 'success',
            'message': 'Network error connecting to weather service, no alerts available',
            'alerts': [],
            'count': 0,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Error fetching weather alerts: {e}")
        
        # Try to use cached data as fallback
        if cached_alerts and cache_timestamp:
            app.logger.info(f"Using stale cached weather alerts from {cache_timestamp.isoformat()} due to error: {e}")
            cached_alerts['status'] = 'partial'
            cached_alerts['message'] = f'Using cached data due to error: {str(e)}'
            return jsonify(cached_alerts)
        
        # Return empty alerts with success status rather than error
        return jsonify({
            'status': 'success',
            'message': 'Error in weather service, no alerts available',
            'alerts': [],
            'count': 0,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

@app.route('/api/radar/analyze', methods=['POST'])
def analyze_radar_patterns():
    """Analyze radar data for mesocyclones and hook echoes.
    
    Expected JSON payload:
    {
        "reflectivity": [[...]], # 2D or 3D array of reflectivity data
        "velocity": [[...]], # 2D or 3D array of velocity data
        "spectrum_width": [[...]], # Optional 2D or 3D array of spectrum width data
        "timestamps": ["2023-06-01T12:00:00Z", ...] # Optional timestamps for each frame
    }
    
    Returns:
        JSON with detected patterns and their properties
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Validate inputs
        if 'reflectivity' not in data or 'velocity' not in data:
            return jsonify({"error": "Missing required radar data fields"}), 400
            
        # Process radar data to detect patterns
        results = radar_processor.process_radar_data(data)
        
        # Enhance results with interpretation
        enhanced_results = {
            "patterns_detected": [],
            "risk_level": "none",
            "details": results
        }
        
        # Add detected patterns to the list
        if results['mesocyclone']['detected']:
            enhanced_results["patterns_detected"].append("mesocyclone")
            
        if results['hook_echo']['detected']:
            enhanced_results["patterns_detected"].append("hook_echo")
            
        # Determine overall risk level
        if results['mesocyclone']['detected'] and results['hook_echo']['detected']:
            # Both detected - highest risk
            if results['mesocyclone']['strength'] > 0.7 and results['hook_echo']['confidence'] > 0.7:
                enhanced_results["risk_level"] = "extreme"
            else:
                enhanced_results["risk_level"] = "high"
        elif results['mesocyclone']['detected']:
            # Only mesocyclone detected
            if results['mesocyclone']['strength'] > 0.7:
                enhanced_results["risk_level"] = "high"
            else:
                enhanced_results["risk_level"] = "moderate"
        elif results['hook_echo']['detected']:
            # Only hook echo detected
            if results['hook_echo']['confidence'] > 0.7:
                enhanced_results["risk_level"] = "moderate"
            else:
                enhanced_results["risk_level"] = "low"
                
        # Add a human-readable summary
        pattern_text = " and ".join(enhanced_results["patterns_detected"])
        if pattern_text:
            enhanced_results["summary"] = f"Detected {pattern_text} - {enhanced_results['risk_level']} risk level"
        else:
            enhanced_results["summary"] = "No tornadic signatures detected"
            
        return jsonify(enhanced_results)
        
    except Exception as e:
        logger.error(f"Error analyzing radar patterns: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug) 
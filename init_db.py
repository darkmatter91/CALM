#!/usr/bin/env python3
import sqlite3
import uuid
import logging
from datetime import datetime, timedelta, timezone
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database path
DB_PATH = "tornado_predictions.db"

def init_database():
    """Initialize the database schema."""
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

def insert_sample_data():
    """Insert sample prediction data for demonstration purposes."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Add sample data only if the database is empty
        cursor.execute("SELECT COUNT(*) FROM predictions")
        count = cursor.fetchone()[0]
        
        if count > 0:
            logger.info("Database already contains data, skipping sample data insertion")
            return True
            
        # Generate some sample locations
        locations = [
            {"name": "Oklahoma City, OK", "lat": 35.4676, "lon": -97.5164},
            {"name": "Norman, OK", "lat": 35.2226, "lon": -97.4395},
            {"name": "Tulsa, OK", "lat": 36.1540, "lon": -95.9928},
            {"name": "Wichita, KS", "lat": 37.6872, "lon": -97.3301},
            {"name": "Dallas, TX", "lat": 32.7767, "lon": -96.7970},
            {"name": "Little Rock, AR", "lat": 34.7465, "lon": -92.2896},
            {"name": "Birmingham, AL", "lat": 33.5186, "lon": -86.8104},
            {"name": "Des Moines, IA", "lat": 41.5868, "lon": -93.6250},
            {"name": "Kansas City, MO", "lat": 39.0997, "lon": -94.5786},
            {"name": "Chicago, IL", "lat": 41.8781, "lon": -87.6298}
        ]
        
        # Generate some sample predictions over the last 14 days
        predictions = []
        now = datetime.now(timezone.utc)
        
        # Different risk levels
        risk_levels = ["low", "moderate", "high", "extreme"]
        
        # Different prediction types
        prediction_types = [
            {"is_ai": True, "nws_alert": False},  # AI prediction only
            {"is_ai": False, "nws_alert": True},  # NWS alert only
            {"is_ai": True, "nws_alert": True}    # Both AI and NWS
        ]
        
        # Generate 50 sample predictions
        for i in range(50):
            # Random location
            location = random.choice(locations)
            
            # Random timestamp within the last 14 days
            days_ago = random.randint(0, 14)
            hours_ago = random.randint(0, 24)
            timestamp = (now - timedelta(days=days_ago, hours=hours_ago)).isoformat()
            
            # Random prediction details
            risk_level = random.choice(risk_levels)
            pred_type = random.choice(prediction_types)
            
            # Risk-appropriate formation chance
            if risk_level == "low":
                formation_chance = random.randint(1, 25)
            elif risk_level == "moderate":
                formation_chance = random.randint(25, 50)
            elif risk_level == "high":
                formation_chance = random.randint(50, 75)
            else:  # extreme
                formation_chance = random.randint(75, 99)
                
            # Weather metrics based on risk level
            if risk_level == "low":
                cape = random.randint(100, 500)
                helicity = random.randint(10, 100)
                shear = f"{random.randint(5, 20)} knots"
            elif risk_level == "moderate":
                cape = random.randint(500, 1500)
                helicity = random.randint(100, 200)
                shear = f"{random.randint(20, 35)} knots"
            elif risk_level == "high":
                cape = random.randint(1500, 3000)
                helicity = random.randint(200, 300)
                shear = f"{random.randint(35, 50)} knots"
            else:  # extreme
                cape = random.randint(3000, 5000)
                helicity = random.randint(300, 500)
                shear = f"{random.randint(50, 70)} knots"
            
            # Generate unique ID
            pred_id = str(uuid.uuid4())
            
            # Create prediction record
            prediction = {
                "id": pred_id,
                "timestamp": now.isoformat(),
                "prediction_time": timestamp,
                "lat": location["lat"],
                "lon": location["lon"],
                "location": location["name"],
                "risk_level": risk_level,
                "formation_chance": formation_chance,
                "cape": cape,
                "helicity": helicity,
                "shear": shear,
                "is_ai_prediction": pred_type["is_ai"],
                "nws_alert": pred_type["nws_alert"],
                "validated": days_ago > 1  # Only validate predictions older than 1 day
            }
            
            predictions.append(prediction)
        
        # Insert the predictions
        for pred in predictions:
            cursor.execute('''
            INSERT INTO predictions (
                id, timestamp, prediction_time, lat, lon, location, 
                risk_level, formation_chance, cape, helicity, shear, 
                is_ai_prediction, nws_alert, validated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pred["id"], pred["timestamp"], pred["prediction_time"], 
                pred["lat"], pred["lon"], pred["location"], 
                pred["risk_level"], pred["formation_chance"], pred["cape"], 
                pred["helicity"], pred["shear"], pred["is_ai_prediction"], 
                pred["nws_alert"], pred["validated"]
            ))
            
            # If the prediction is validated, create a validation result
            if pred["validated"]:
                # 70% chance of being correct for high/extreme risks
                # 50% chance of being correct for moderate risks
                # 30% chance of being correct for low risks
                correct_probability = 0.3
                if pred["risk_level"] == "moderate":
                    correct_probability = 0.5
                elif pred["risk_level"] in ["high", "extreme"]:
                    correct_probability = 0.7
                    
                was_correct = random.random() < correct_probability
                
                # Create validation record
                validation_time = (datetime.fromisoformat(pred["prediction_time"]) + timedelta(days=1)).isoformat()
                distance_error = random.randint(5, 150) if was_correct else random.randint(100, 300)
                
                if was_correct:
                    if distance_error <= 50:
                        notes = f"Tornado occurred {distance_error}km from prediction, within expected radius."
                    else:
                        notes = f"Tornado occurred {distance_error}km from prediction, slightly outside expected radius."
                else:
                    notes = "No tornado activity detected in the area within the prediction window."
                
                cursor.execute('''
                INSERT INTO validation_results (
                    prediction_id, validated_at, was_correct, actual_event_id,
                    distance_error_km, time_error_minutes, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pred["id"], validation_time, was_correct, 
                    str(uuid.uuid4()) if was_correct else None,
                    distance_error if was_correct else None,
                    random.randint(10, 120) if was_correct else None,
                    notes
                ))
        
        conn.commit()
        logger.info(f"Inserted {len(predictions)} sample predictions")
        return True
    except Exception as e:
        logger.error(f"Error inserting sample data: {e}")
        return False
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    logger.info("Initializing database...")
    if init_database():
        logger.info("Inserting sample data...")
        if insert_sample_data():
            logger.info("Database setup complete!")
        else:
            logger.error("Failed to insert sample data")
    else:
        logger.error("Failed to initialize database") 
# CALM - Climate Assessment & Logging Monitor

![Main Page](/assets/calmMainPage.png)

## Overview

CALM (Climate Assessment & Logging Monitor) is a state-of-the-art web application that uses machine learning and real-time meteorological data to predict tornado formation probability. By combining traditional weather data analysis with advanced AI techniques, CALM provides early warnings and risk assessments to help enhance public safety during severe weather events.

![Tornado Predictions](/assets/calmTornado.png)

## Vision

The vision for CALM is to create an accessible, accurate, and real-time tornado prediction system that:

- Democratizes access to sophisticated meteorological analysis through a user-friendly interface
- Provides early warnings to communities that might be affected by tornado activity
- Improves prediction accuracy through continuous learning and data validation
- Integrates seamlessly with official weather alerts while providing additional AI-powered insights
- Serves as both an educational tool and a practical safety resource

## Features

- **Real-time Tornado Predictions**: AI-generated predictions with advanced risk assessment
- **Interactive Risk Visualization**: Color-coded map interface showing tornado risk levels
- **Live Weather Alert Integration**: Direct integration with National Weather Service alerts
- **Interactive Radar Overlays**: Multiple radar data sources including NOAA/NWS, RainViewer, and OpenWeatherMap with refresh capability
- **Multiple Basemap Options**: Choose between Street Map, Satellite, or Topographic views
- **Weather Alert Filtering**: Filter active NWS alerts by type (Tornado Warning, Tornado Watch, etc.)
- **Organized Legend Display**: Categorized legend showing AI Alerts and NWS Alerts separately
- **Continuous Learning**: Model that improves over time through validation and feedback
- **Multi-factor Analysis**: Incorporates CAPE values, wind shear, helicity, and historical patterns
- **Responsive Design**: Works seamlessly on desktop devices
- **HTTPS Support**: Secure communications with SSL/TLS encryption for enhanced security
- **AI-Only Prediction Panel**: Dedicated panel showing only AI model predictions, separate from NWS alerts
- **API Response Caching**: Intelligent caching system to prevent rate limit issues with external APIs
- **Pattern Recognition**: Automated detection of mesocyclones and hook echoes in radar imagery

## Coming Soon

- **Path Prediction**: More accurate forecasting of potential tornado tracks
- **Impact Assessment**: Estimation of potential damage based on predicted storm intensity
- **Mobile Browsing**: Native mobile web browsing experience with better formatting and web gui
- **Advanced NEXRAD Integration**: Enhanced integration with the latest NEXRAD/NOAA radar data

## AI Model Architecture

CALM leverages a sophisticated neural network architecture to predict tornado formation:

### Data Collection & Processing

1. **Input Data Sources**:
   - Real-time meteorological measurements (CAPE, wind shear, helicity)
   - NWS alert feed integration
   - Historical tornado frequency data
   - Radar imagery analysis
   - Seasonal and diurnal patterns

2. **Feature Engineering**:
   - Thermodynamic indices calculation
   - Kinematic parameter normalization
   - Temporal pattern analysis
   - Geographic risk profiling

### Neural Network Structure

- **Input Layer**: 12 meteorological and geographical features
- **Hidden Layers**: Dense layers with ReLU activation and dropout regularization
- **Attention Mechanism**: Focuses on the most relevant meteorological factors
- **Output Layer**: Sigmoid activation for probability and softmax for risk classification

### Training & Adaptation

- **Initial Training**: Historical tornado events with balanced positive/negative examples
- **Continuous Learning**: Adjusts weights based on prediction performance
- **Validation Pipeline**: Automatically verifies predictions against actual tornado reports
- **Meteorological Constraints**: Domain knowledge rules to reduce false positives

Current model accuracy: 87.3% (up from 78% at initial deployment)

## Radar Pattern Recognition

The system now incorporates computer vision algorithms to detect key tornadic signatures in radar data:

1. **Mesocyclone Detection**:
   - Analyzes velocity data to identify rotating storm structures
   - Calculates rotation strength, diameter, and location
   - Optimized for increased sensitivity to weaker rotation signatures
   - Uses velocity gradient analysis with lower thresholds to detect early stage rotation

2. **Hook Echo Detection**:
   - Identifies hook-shaped appendages in reflectivity data
   - Analyzes contour curvature and convexity
   - Enhanced detection of smaller and less defined hook echoes
   - Calculates confidence scores for detected hook echoes

3. **Weak Rotation Detection**:
   - Detects subtle rotation patterns in velocity data
   - Provides early warnings for developing mesocyclones
   - Flags rotation signatures below mesocyclone threshold but still potentially significant

4. **Combined Risk Assessment**:
   - Evaluates the presence of multiple patterns to determine overall risk level
   - Assigns risk categories: none, low, moderate, high, or extreme
   - Uses lower thresholds for NWS alert validation
   - Provides human-readable summaries of detected patterns

The pattern detection system is calibrated to favor higher sensitivity (more potential false positives) over missing actual tornado signatures, as this better aligns with the system's safety-oriented objectives.

## Installation & Setup

### Prerequisites

- Python 3.9+
- pip (Python package manager)
- Git

### Method 1: Local Installation

1. Clone the repository:
```bash
git clone https://github.com/darkmatter91/CALM.git
cd CALM
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
# or
flask run
```

### Method 2: Docker Installation with HTTPS

1. Clone the repository:
```bash
git clone https://github.com/darkmatter91/CALM.git
cd CALM
```

2. Generate SSL certificates (for development purposes):
```bash
mkdir -p ./nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout ./nginx/ssl/nginx.key -out ./nginx/ssl/nginx.crt -subj "/CN=localhost"
```

3. Build and run with Docker Compose:
```bash
docker-compose up --build
```

4. Access the application:
- Open a browser and navigate to `https://localhost`
- For HTTP, use `http://localhost` (will redirect to HTTPS)

#### Docker Commands Reference

| Command | Description |
|---------|-------------|
| `docker-compose up` | Start the application |
| `docker-compose up -d` | Start in detached mode (background) |
| `docker-compose down` | Stop the application |
| `docker-compose logs -f` | View logs |
| `docker-compose restart` | Restart the application |

### Docker Dependencies

The Dockerfile includes all necessary dependencies for running the application:

- **Basic Requirements**: Python libraries and core dependencies
- **Geospatial Libraries**: Support for map projections and geographic calculations
- **OpenCV Dependencies**: Libraries required for computer vision and radar analysis:
  - `libgl1-mesa-glx`: OpenGL library needed by OpenCV
  - `libglib2.0-0`, `libsm6`, `libxext6`, `libxrender-dev`: X11/GUI support libraries

If you encounter the error `ImportError: libGL.so.1: cannot open shared object file`, make sure your Docker build includes the OpenCV dependencies listed above.

### Production HTTPS Setup

For production deployments, follow these steps to use proper SSL certificates:

1. Obtain SSL certificates from a Certificate Authority (like Let's Encrypt):
```bash
# Install certbot
apt-get update
apt-get install certbot

# Generate certificates for your domain
certbot certonly --standalone -d yourdomain.com -d www.yourdomain.com
```

2. Copy the certificates to your NGINX ssl directory:
```bash
cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem ./nginx/ssl/nginx.crt
cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ./nginx/ssl/nginx.key
```

3. Update `nginx/nginx.conf` server_name to match your domain:
```
server_name yourdomain.com www.yourdomain.com;
```

4. Build and run the application:
```bash
docker-compose up --build -d
```

Note: Remember to set up automatic renewal for Let's Encrypt certificates as they expire every 90 days.

## API Endpoints

### Web Pages
- `GET /` - Main dashboard
- `GET /tornado` - Tornado prediction map interface
- `GET /model-stats` - Model statistics and performance metrics

### Data Endpoints
- `GET /api/tornado/predictions` - Current tornado predictions (AI only)
- `GET /api/weather/alerts` - Active weather alerts from NWS
- `POST /api/predict` - Submit location for specific prediction
- `POST /api/radar` - Process radar data and detect patterns
- `POST /api/radar/analyze` - Analyze radar data for mesocyclones and hook echoes

## Data Sources

CALM integrates data from multiple free, public APIs:

1. **National Weather Service (NWS) API**
   - Official US weather forecasts and alerts
   - No API key required

2. **Nominatim API**
   - Geocoding service for location lookups
   - Rate-limited, no API key required

3. **Open-Meteo API**
   - Global weather data
   - No API key required, responses cached to prevent rate limiting

4. **RainViewer Radar API**
   - Real-time precipitation radar data
   - No API key required

5. **OpenWeatherMap**
   - Alternative precipitation data source
   - Basic features available without API key

## Frequently Asked Questions (FAQ)

### General Questions

#### Q: What is CALM and how does it work?
A: CALM (Climate Assessment & Logging Monitor) is an AI-powered tornado prediction system that analyzes meteorological data, radar imagery, and historical patterns to estimate tornado formation probability. It combines traditional weather metrics with machine learning to provide risk assessments and early warnings.

#### Q: How accurate are the AI predictions?
A: The AI model currently has an accuracy of approximately 87.3%. However, tornado prediction is inherently challenging, and the system is designed to err on the side of caution with higher sensitivity. This means it may sometimes show elevated risk levels in borderline conditions. The model continuously improves through validation against actual tornado occurrences.

#### Q: What data does CALM use to make predictions?
A: CALM uses multiple data sources including real-time meteorological measurements (CAPE, helicity, wind shear), live radar imagery, historical tornado patterns, and official NWS alerts. These inputs are processed through our neural network to generate risk assessments.

### Prediction Interpretation

#### Q: Why do I see a tornado prediction when there's nothing on the radar?
A: Tornado formation depends on atmospheric conditions that may exist before visible storm development. The system detects favorable environmental conditions (high CAPE, strong wind shear, elevated helicity) that could support tornado development in the coming hours, even if no precipitation is currently visible on radar. Think of it as a "potential for development" indicator rather than a "tornado is currently forming" alert.

#### Q: What do the different risk levels mean?
A: The risk levels indicate the probability of tornado formation:
- **Very Low (< 10%)**: Minimal tornado risk but conditions are worth monitoring
- **Low (10-29%)**: Some favorable conditions exist but tornado formation is not likely
- **Moderate (30-49%)**: Several favorable conditions exist creating notable tornado potential
- **High (50-69%)**: Many favorable conditions exist creating significant tornado potential
- **Extreme (70%+)**: Nearly all conditions favor tornado formation with imminent threat

#### Q: Why can predictions change quickly?
A: Atmospheric conditions are dynamic and can change rapidly. The system updates predictions as new data becomes available, which may result in risk levels increasing or decreasing as conditions evolve.

### Map and Interface

#### Q: What's the difference between NWS alerts and AI predictions?
A: NWS (National Weather Service) alerts are official government warnings issued by meteorologists when severe weather is imminent or occurring. AI predictions are automated risk assessments generated by our machine learning model based on current conditions. The AI predictions may appear before official alerts and cover areas not yet under official warnings.

#### Q: Why do alerts sometimes disappear when I refresh the page?
A: This may happen if there's a timing issue with the API connection. Using the "All" filter button or the refresh button should restore the alerts. We've implemented automatic retries to minimize this issue.

#### Q: How often does the data update?
A: The system refreshes data approximately every 5 minutes for both AI predictions and NWS alerts. The countdown timer in the bottom-left corner shows the time until the next update.

#### Q: What radar sources are available and which should I use?
A: CALM offers three radar sources:
- **RainViewer**: Most reliable and generally fastest (default)
- **OpenWeatherMap**: Alternative source with good coverage
- **NOAA Radar**: Official source but may be rate-limited or slower

You can switch between them using the layer control in the bottom-right corner of the map.

### Technical Issues

#### Q: Why am I seeing "Error fetching alerts" messages?
A: This usually happens if the server is still initializing or there's a temporary connection issue. You can click the "Retry" button or the "All" filter button to refresh the alerts. The system also automatically retries loading alerts after a few seconds.

#### Q: The map zoom behavior seems inconsistent - why?
A: We've updated the system to maintain your current view rather than automatically zooming to alerts. You can use the "Reset View" button (globe icon) in the bottom-right corner to return to the full US view at any time.

#### Q: How can I report bugs or suggest features?
A: Please submit issues to our GitHub repository or contact us through the About page. We appreciate your feedback and are continuously working to improve the system.

## Technical Stack

- **Backend**: Flask (Python)
- **Machine Learning**: TensorFlow, scikit-learn
- **Computer Vision**: OpenCV, scikit-image
- **Visualization**: Leaflet.js, Chart.js
- **Frontend**: Bootstrap, HTML5, CSS3, JavaScript
- **Data Processing**: NumPy, Pandas, SciPy
- **Security**: HTTPS with SSL/TLS via Nginx reverse proxy

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 

## Disclaimer

**IMPORTANT: CALM is provided for educational and informational purposes only.**

This software and its tornado predictions are not a replacement for official National Weather Service (NWS) or government-issued warnings and alerts. The creators, developers, and contributors of CALM:

- Are not liable for any damages, injuries, or losses that may result from using or relying on this software
- Do not guarantee the accuracy, completeness, or timeliness of any predictions or notifications
- Are not responsible for any actions taken or not taken based on the information provided by this application
- Make no warranty that the service will meet your requirements or be available on an uninterrupted, secure, or error-free basis

Always rely on official government weather services for critical weather safety information and follow local emergency management instructions during severe weather events.

By using CALM, you acknowledge and agree to these terms. 

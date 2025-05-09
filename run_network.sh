#!/bin/bash

# Get the IP address of this machine
IP_ADDRESS=$(hostname -I | awk '{print $1}')

echo "Starting CALM Weather Application with network access"
echo "Your IP address is: $IP_ADDRESS"
echo "-------------------------------------------"
echo "Access the application from other devices at:"
echo "Frontend: http://$IP_ADDRESS:3000"
echo "Backend API: http://$IP_ADDRESS:5000"
echo "-------------------------------------------"

# Start the Flask backend in the background
echo "Starting Flask backend..."
cd /home/darkma773r/Documents/GitHub/CALM
source venv/bin/activate
export FLASK_APP=app.py
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=5000 &
FLASK_PID=$!

# Wait a moment for Flask to start
sleep 2

# Start the React frontend
echo "Starting React frontend..."
cd /home/darkma773r/Documents/GitHub/CALM/frontend
npm start -- --host 0.0.0.0

# When the React app is stopped, also stop the Flask app
kill $FLASK_PID
echo "Application stopped." 
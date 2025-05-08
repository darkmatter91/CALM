/**
 * CALM (Climate Assessment & Logging Monitor) - Location Search functionality
 * Handles ZIP code search and location-based tornado predictions
 */

// Function to search for a location by zipcode
function searchLocation(zipcode) {
    console.log("Searching for zipcode:", zipcode);
    
    // Validate zipcode format
    if(!zipcode || !zipcode.match(/^\d{5}$/)) {
        showError("Please enter a valid 5-digit US ZIP code");
        return;
    }
    
    // Hide any previous error
    hideError();
    
    // Show loading state in the predictions panel
    showPredictionsLoading();
    
    // Make API request to get prediction for this location
    fetch(`/api/predict?zipcode=${zipcode}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Server responded with status ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Received prediction data:", data);
            
            // Center map on the location
            if (data.location && data.location.lat && data.location.lon) {
                if (map) {
                    map.setView([data.location.lat, data.location.lon], 8);
                }
            }
            
            // Process and display the prediction
            if (data.predictions && data.predictions.length > 0) {
                displayPredictions(data.predictions);
                updatePredictionCount(data.predictions.length);
            } else {
                displayNoPredictionsMessage();
            }
            
            // Hide loading state
            hidePredictionsLoading();
        })
        .catch(error => {
            console.error("Error fetching prediction:", error);
            showError("Failed to get prediction. Please try again later.");
            hidePredictionsLoading();
        });
}

/**
 * Show loading state in the predictions panel
 */
function showPredictionsLoading() {
    const loadingElement = document.getElementById('predictionsLoading');
    if (loadingElement) {
        loadingElement.style.display = 'block';
    }
    
    // Hide other prediction content while loading
    const noPredictions = document.getElementById('no-predictions');
    const predictionsList = document.getElementById('predictions-list');
    
    if (noPredictions) noPredictions.style.display = 'none';
    if (predictionsList) predictionsList.style.display = 'none';
    
    // Start progress simulation if it exists
    if (typeof simulateProgress === 'function') {
        simulateProgress();
    }
}

/**
 * Hide loading state in the predictions panel
 */
function hidePredictionsLoading() {
    const loadingElement = document.getElementById('predictionsLoading');
    if (loadingElement) {
        loadingElement.style.display = 'none';
    }
    
    // Stop progress simulation if it's running
    if (window.progressInterval) {
        clearInterval(window.progressInterval);
    }
}

/**
 * Show error message to the user
 * @param {string} message - The error message to display
 */
function showError(message) {
    const errorElement = document.getElementById('errorMessage');
    if (errorElement) {
        errorElement.textContent = message;
        errorElement.style.display = 'block';
    }
}

/**
 * Hide the error message
 */
function hideError() {
    const errorElement = document.getElementById('errorMessage');
    if (errorElement) {
        errorElement.style.display = 'none';
    }
} 
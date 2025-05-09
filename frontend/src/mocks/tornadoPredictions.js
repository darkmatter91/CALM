// Mock tornado prediction data for development and testing
const mockTornadoPredictions = {
  predictions: [
    {
      id: 1,
      lat: 35.6895,
      lon: -97.3281,
      location: "Oklahoma City, OK",
      risk_level: "high",
      formation_chance: 85,
      prediction_time: "2025-05-09T12:30:00Z",
      cape: 2500,
      helicity: 350,
      shear: "strong",
      nws_alert: true
    },
    {
      id: 2,
      lat: 33.4484,
      lon: -94.0418,
      location: "Texarkana, TX",
      risk_level: "medium",
      formation_chance: 65,
      prediction_time: "2025-05-09T12:30:00Z",
      cape: 1800,
      helicity: 220,
      shear: "moderate",
      nws_alert: false
    },
    {
      id: 3,
      lat: 37.7749,
      lon: -90.3858,
      location: "Farmington, MO",
      risk_level: "low",
      formation_chance: 35,
      prediction_time: "2025-05-09T12:30:00Z",
      cape: 1200,
      helicity: 150,
      shear: "light",
      nws_alert: false
    }
  ],
  status: "success"
};

export default mockTornadoPredictions; 
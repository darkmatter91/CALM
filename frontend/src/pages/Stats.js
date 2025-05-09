import React, { useState, useEffect } from 'react';
import { Bar, Line, Pie } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import weatherService from '../services/api';
import Loader from '../components/Loader';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const Stats = () => {
  const [stats, setStats] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [timeRange, setTimeRange] = useState('month'); // 'week', 'month', 'year'

  useEffect(() => {
    const fetchStats = async () => {
      try {
        setIsLoading(true);
        const data = await weatherService.getPredictionStats({ range: timeRange });
        setStats(data);
      } catch (err) {
        setError('Failed to load prediction statistics. Please try again later.');
        console.error(err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchStats();
  }, [timeRange]);

  const handleTimeRangeChange = (range) => {
    setTimeRange(range);
  };

  if (isLoading) {
    return <Loader message="Loading statistics..." />;
  }

  if (error) {
    return (
      <div className="container py-5">
        <div className="alert alert-danger">{error}</div>
        <button 
          className="btn btn-primary" 
          onClick={() => window.location.reload()}
        >
          Retry
        </button>
      </div>
    );
  }

  // Prepare chart data if stats are available
  const prepareAccuracyChartData = () => {
    if (!stats || !stats.accuracy_over_time) return null;

    const labels = stats.accuracy_over_time.map(entry => entry.date);
    const accuracyData = stats.accuracy_over_time.map(entry => entry.accuracy * 100);

    return {
      labels,
      datasets: [
        {
          label: 'Prediction Accuracy (%)',
          data: accuracyData,
          fill: true,
          backgroundColor: 'rgba(59, 130, 246, 0.2)',
          borderColor: 'rgba(59, 130, 246, 1)',
          tension: 0.4
        }
      ]
    };
  };

  const prepareRiskDistributionData = () => {
    if (!stats || !stats.risk_distribution) return null;

    const labels = Object.keys(stats.risk_distribution);
    const data = Object.values(stats.risk_distribution);

    return {
      labels,
      datasets: [
        {
          label: 'Risk Level Distribution',
          data,
          backgroundColor: [
            'rgba(16, 185, 129, 0.7)',  // Green for low
            'rgba(245, 158, 11, 0.7)',  // Yellow for medium
            'rgba(239, 68, 68, 0.7)',   // Red for high
            'rgba(127, 29, 29, 0.7)'    // Dark red for extreme
          ],
          borderColor: [
            'rgba(16, 185, 129, 1)',
            'rgba(245, 158, 11, 1)',
            'rgba(239, 68, 68, 1)',
            'rgba(127, 29, 29, 1)'
          ],
          borderWidth: 1
        }
      ]
    };
  };

  const prepareValidationResultsData = () => {
    if (!stats || !stats.validation_results) return null;

    return {
      labels: ['Correct', 'Incorrect', 'Pending'],
      datasets: [
        {
          label: 'Validation Results',
          data: [
            stats.validation_results.correct || 0,
            stats.validation_results.incorrect || 0,
            stats.validation_results.pending || 0
          ],
          backgroundColor: [
            'rgba(16, 185, 129, 0.7)',  // Green for correct
            'rgba(239, 68, 68, 0.7)',   // Red for incorrect
            'rgba(156, 163, 175, 0.7)'  // Gray for pending
          ],
          borderColor: [
            'rgba(16, 185, 129, 1)',
            'rgba(239, 68, 68, 1)',
            'rgba(156, 163, 175, 1)'
          ],
          borderWidth: 1
        }
      ]
    };
  };

  const prepareErrorDistanceData = () => {
    if (!stats || !stats.error_distribution) return null;

    const labels = Object.keys(stats.error_distribution.distance);
    const data = Object.values(stats.error_distribution.distance);

    return {
      labels,
      datasets: [
        {
          label: 'Distance Error Distribution (km)',
          data,
          backgroundColor: 'rgba(139, 92, 246, 0.7)',
          borderColor: 'rgba(139, 92, 246, 1)',
          borderWidth: 1
        }
      ]
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#fff',
        bodyColor: '#fff',
        borderColor: 'rgba(255, 255, 255, 0.1)',
        borderWidth: 1
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        },
        ticks: {
          color: 'rgba(255, 255, 255, 0.7)'
        }
      },
      x: {
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        },
        ticks: {
          color: 'rgba(255, 255, 255, 0.7)'
        }
      }
    }
  };

  return (
    <div className="stats-container">
      <div className="row mb-4">
        <div className="col">
          <h2 className="mb-3">Model Performance Statistics</h2>
          <p className="text-muted">
            These stats show our model's prediction accuracy, validation results, and error distribution.
          </p>

          <div className="btn-group mb-4">
            <button 
              className={`btn ${timeRange === 'week' ? 'btn-primary' : 'btn-outline-primary'}`} 
              onClick={() => handleTimeRangeChange('week')}
            >
              Last Week
            </button>
            <button 
              className={`btn ${timeRange === 'month' ? 'btn-primary' : 'btn-outline-primary'}`} 
              onClick={() => handleTimeRangeChange('month')}
            >
              Last Month
            </button>
            <button 
              className={`btn ${timeRange === 'year' ? 'btn-primary' : 'btn-outline-primary'}`} 
              onClick={() => handleTimeRangeChange('year')}
            >
              Last Year
            </button>
          </div>
        </div>
      </div>

      <div className="row mb-4">
        <div className="col-md-6 mb-4">
          <div className="card h-100">
            <div className="card-body">
              <h5 className="card-title">Prediction Accuracy Over Time</h5>
              <div style={{ height: '300px' }}>
                {stats && stats.accuracy_over_time ? (
                  <Line data={prepareAccuracyChartData()} options={chartOptions} />
                ) : (
                  <div className="d-flex align-items-center justify-content-center h-100">
                    <p className="text-muted">No accuracy data available</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
        
        <div className="col-md-6 mb-4">
          <div className="card h-100">
            <div className="card-body">
              <h5 className="card-title">Risk Level Distribution</h5>
              <div style={{ height: '300px' }}>
                {stats && stats.risk_distribution ? (
                  <Pie data={prepareRiskDistributionData()} options={chartOptions} />
                ) : (
                  <div className="d-flex align-items-center justify-content-center h-100">
                    <p className="text-muted">No risk distribution data available</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="row mb-4">
        <div className="col-md-6 mb-4">
          <div className="card h-100">
            <div className="card-body">
              <h5 className="card-title">Validation Results</h5>
              <div style={{ height: '300px' }}>
                {stats && stats.validation_results ? (
                  <Pie data={prepareValidationResultsData()} options={chartOptions} />
                ) : (
                  <div className="d-flex align-items-center justify-content-center h-100">
                    <p className="text-muted">No validation results available</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
        
        <div className="col-md-6 mb-4">
          <div className="card h-100">
            <div className="card-body">
              <h5 className="card-title">Distance Error Distribution</h5>
              <div style={{ height: '300px' }}>
                {stats && stats.error_distribution ? (
                  <Bar data={prepareErrorDistanceData()} options={chartOptions} />
                ) : (
                  <div className="d-flex align-items-center justify-content-center h-100">
                    <p className="text-muted">No error distribution data available</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="row">
        <div className="col-12">
          <div className="card">
            <div className="card-body">
              <h5 className="card-title">Overall Performance Metrics</h5>
              <div className="row">
                <div className="col-md-3 mb-3">
                  <div className="text-center p-3">
                    <h3 className="text-primary mb-0">{stats?.overall_accuracy ? (stats.overall_accuracy * 100).toFixed(1) : 'N/A'}%</h3>
                    <p className="text-muted">Overall Accuracy</p>
                  </div>
                </div>
                <div className="col-md-3 mb-3">
                  <div className="text-center p-3">
                    <h3 className="text-success mb-0">{stats?.total_predictions || 'N/A'}</h3>
                    <p className="text-muted">Total Predictions</p>
                  </div>
                </div>
                <div className="col-md-3 mb-3">
                  <div className="text-center p-3">
                    <h3 className="text-warning mb-0">{stats?.avg_distance_error ? stats.avg_distance_error.toFixed(1) : 'N/A'} km</h3>
                    <p className="text-muted">Avg. Distance Error</p>
                  </div>
                </div>
                <div className="col-md-3 mb-3">
                  <div className="text-center p-3">
                    <h3 className="text-danger mb-0">{stats?.avg_time_error ? stats.avg_time_error.toFixed(1) : 'N/A'} min</h3>
                    <p className="text-muted">Avg. Time Error</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Stats; 
import React, { useState, useEffect } from 'react';
import weatherService from '../services/api';
import Loader from '../components/Loader';

const Model = () => {
  const [modelInfo, setModelInfo] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingParams, setTrainingParams] = useState({
    epochs: 10,
    batch_size: 32,
    learning_rate: 0.001,
    validation_split: 0.2
  });

  useEffect(() => {
    const fetchModelStatus = async () => {
      try {
        const data = await weatherService.getModelStatus();
        setModelInfo(data);
      } catch (err) {
        setError('Failed to load model information. Please try again later.');
        console.error(err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchModelStatus();

    // Poll for model status updates every 10 seconds if training
    const interval = setInterval(() => {
      if (isTraining) {
        fetchModelStatus();
      }
    }, 10000);

    return () => clearInterval(interval);
  }, [isTraining]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setTrainingParams({
      ...trainingParams,
      [name]: name === 'epochs' || name === 'batch_size' ? parseInt(value, 10) : parseFloat(value)
    });
  };

  const handleTrainModel = async (e) => {
    e.preventDefault();
    try {
      setIsTraining(true);
      await weatherService.trainModel(trainingParams);
      // Training started, now we'll poll for updates
    } catch (err) {
      setError('Failed to start model training. Please try again.');
      setIsTraining(false);
      console.error(err);
    }
  };

  if (isLoading) {
    return <Loader message="Loading model information..." />;
  }

  return (
    <div className="model-container">
      <div className="row mb-4">
        <div className="col-12">
          <h2 className="mb-3">AI Model Information</h2>
          <p className="text-muted">
            Details about our tornado prediction neural network and its current training status.
          </p>
        </div>
      </div>

      {error && (
        <div className="alert alert-danger mb-4">{error}</div>
      )}

      <div className="row mb-4">
        <div className="col-md-8">
          <div className="card">
            <div className="card-body">
              <h5 className="card-title">Model Architecture</h5>
              <div className="model-architecture">
                {modelInfo?.architecture ? (
                  <pre className="bg-dark p-3 rounded" style={{ color: 'var(--text)', maxHeight: '400px', overflow: 'auto' }}>
                    {JSON.stringify(modelInfo.architecture, null, 2)}
                  </pre>
                ) : (
                  <p className="text-muted">No architecture information available</p>
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="col-md-4">
          <div className="card mb-4">
            <div className="card-body">
              <h5 className="card-title">Model Status</h5>
              <div className="model-status">
                <div className="d-flex align-items-center mb-3">
                  <div className={`status-indicator ${modelInfo?.status === 'ready' ? 'bg-success' : modelInfo?.status === 'training' ? 'bg-warning' : 'bg-danger'}`} style={{ width: 12, height: 12, borderRadius: '50%', marginRight: 10 }}></div>
                  <span className="text-capitalize">{modelInfo?.status || 'Unknown'}</span>
                </div>
                <p><strong>Last Trained:</strong> {modelInfo?.last_trained ? new Date(modelInfo.last_trained).toLocaleString() : 'Never'}</p>
                <p><strong>Current Epoch:</strong> {modelInfo?.training_progress?.current_epoch || 0}/{modelInfo?.training_progress?.total_epochs || 0}</p>
                <p><strong>Accuracy:</strong> {modelInfo?.metrics?.accuracy ? (modelInfo.metrics.accuracy * 100).toFixed(2) + '%' : 'N/A'}</p>
                <p><strong>Loss:</strong> {modelInfo?.metrics?.loss?.toFixed(4) || 'N/A'}</p>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-body">
              <h5 className="card-title">Training Progress</h5>
              {modelInfo?.status === 'training' ? (
                <div>
                  <div className="progress mb-3">
                    <div 
                      className="progress-bar progress-bar-striped progress-bar-animated" 
                      role="progressbar" 
                      style={{ width: `${modelInfo?.training_progress ? (modelInfo.training_progress.current_epoch / modelInfo.training_progress.total_epochs * 100) : 0}%` }}
                      aria-valuenow={modelInfo?.training_progress?.current_epoch || 0} 
                      aria-valuemin="0" 
                      aria-valuemax={modelInfo?.training_progress?.total_epochs || 0}
                    ></div>
                  </div>
                  <p className="text-center text-muted">Training in progress...</p>
                </div>
              ) : (
                <form onSubmit={handleTrainModel}>
                  <div className="mb-3">
                    <label htmlFor="epochs" className="form-label">Epochs</label>
                    <input
                      type="number"
                      className="form-control"
                      id="epochs"
                      name="epochs"
                      min="1"
                      max="100"
                      value={trainingParams.epochs}
                      onChange={handleInputChange}
                    />
                  </div>
                  <div className="mb-3">
                    <label htmlFor="batch_size" className="form-label">Batch Size</label>
                    <input
                      type="number"
                      className="form-control"
                      id="batch_size"
                      name="batch_size"
                      min="1"
                      max="128"
                      value={trainingParams.batch_size}
                      onChange={handleInputChange}
                    />
                  </div>
                  <div className="mb-3">
                    <label htmlFor="learning_rate" className="form-label">Learning Rate</label>
                    <input
                      type="number"
                      className="form-control"
                      id="learning_rate"
                      name="learning_rate"
                      min="0.0001"
                      max="0.1"
                      step="0.0001"
                      value={trainingParams.learning_rate}
                      onChange={handleInputChange}
                    />
                  </div>
                  <button type="submit" className="btn btn-primary w-100">
                    Start Training
                  </button>
                </form>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="row">
        <div className="col-12">
          <div className="card">
            <div className="card-body">
              <h5 className="card-title">Training History</h5>
              {modelInfo?.training_history && modelInfo.training_history.length > 0 ? (
                <div className="table-responsive">
                  <table className="table table-hover">
                    <thead>
                      <tr>
                        <th>Date</th>
                        <th>Epochs</th>
                        <th>Batch Size</th>
                        <th>Final Accuracy</th>
                        <th>Final Loss</th>
                      </tr>
                    </thead>
                    <tbody>
                      {modelInfo.training_history.map((entry, index) => (
                        <tr key={index}>
                          <td>{new Date(entry.date).toLocaleString()}</td>
                          <td>{entry.epochs}</td>
                          <td>{entry.batch_size}</td>
                          <td>{(entry.accuracy * 100).toFixed(2)}%</td>
                          <td>{entry.loss.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="text-muted">No training history available</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Model; 
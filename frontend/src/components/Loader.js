import React from 'react';

const Loader = ({ message = 'Loading...' }) => {
  return (
    <div className="loader-container" style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      height: '100vh',
      backgroundColor: 'var(--bg)',
      color: 'var(--text)'
    }}>
      <div className="spinner-border text-primary mb-3" role="status" style={{ width: '3rem', height: '3rem' }}>
        <span className="visually-hidden">Loading...</span>
      </div>
      <p>{message}</p>
    </div>
  );
};

export default Loader; 
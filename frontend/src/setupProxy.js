const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  // Add middleware to handle source map requests
  app.use(
    '/__source-maps__',
    createProxyMiddleware({
      target: 'http://localhost:3000',
      changeOrigin: true,
      pathRewrite: {
        '^/__source-maps__': '',
      },
      onProxyReq: (proxyReq, req, res) => {
        // For debugging source map requests
        console.log('Source map proxy request:', req.url);
      },
    })
  );
  
  // Forward API requests to your backend server
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'http://localhost:5000',
      changeOrigin: true,
    })
  );
}; 
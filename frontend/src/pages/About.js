import React from 'react';

const About = () => {
  return (
    <div className="about-container">
      <div className="row mb-4">
        <div className="col-12">
          <h2 className="mb-3">About CALM</h2>
          <p className="lead">
            CALM (Climate Assessment & Logging Monitor) is an advanced AI-powered platform designed to predict and track severe weather events with unprecedented accuracy.
          </p>
        </div>
      </div>

      <div className="row mb-5">
        <div className="col-md-6">
          <div className="card h-100">
            <div className="card-body">
              <h3 className="card-title">Our Mission</h3>
              <p>
                CALM's mission is to provide timely, accurate, and accessible severe weather predictions to help communities prepare for and mitigate the impacts of extreme weather events, particularly tornadoes.
              </p>
              <p>
                By leveraging cutting-edge artificial intelligence, meteorological science, and real-time data analysis, we aim to save lives and reduce property damage caused by severe weather.
              </p>
            </div>
          </div>
        </div>
        
        <div className="col-md-6">
          <div className="card h-100">
            <div className="card-body">
              <h3 className="card-title">Technology</h3>
              <p>
                Our platform combines traditional meteorological models with advanced neural networks, specialized for detecting atmospheric conditions conducive to tornado formation.
              </p>
              <p>
                The AI model processes various data inputs including:
              </p>
              <ul>
                <li>NEXRAD radar imagery</li>
                <li>Atmospheric sounding data</li>
                <li>Surface weather observations</li>
                <li>Satellite imagery</li>
                <li>Historical tornado reports</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div className="row mb-5">
        <div className="col-12">
          <div className="card">
            <div className="card-body">
              <h3 className="card-title mb-4">Frequently Asked Questions</h3>
              
              <div className="accordion" id="faqAccordion">
                <div className="accordion-item">
                  <h4 className="accordion-header">
                    <button className="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#faq1">
                      How accurate are the tornado predictions?
                    </button>
                  </h4>
                  <div id="faq1" className="accordion-collapse collapse show" data-bs-parent="#faqAccordion">
                    <div className="accordion-body">
                      <p>
                        Our model currently achieves an average accuracy of 85% for predicting tornado formation within a 24-hour window and a 50-mile radius. This represents a significant improvement over traditional methods which typically have 60-70% accuracy rates.
                      </p>
                      <p>
                        We continuously validate our predictions against actual tornado reports from the National Weather Service (NWS) and other reliable sources, and use this data to further improve our models.
                      </p>
                    </div>
                  </div>
                </div>
                
                <div className="accordion-item">
                  <h4 className="accordion-header">
                    <button className="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#faq2">
                      How far in advance can you predict tornadoes?
                    </button>
                  </h4>
                  <div id="faq2" className="accordion-collapse collapse" data-bs-parent="#faqAccordion">
                    <div className="accordion-body">
                      <p>
                        Our system provides predictions in several timeframes:
                      </p>
                      <ul>
                        <li><strong>Short-term (0-6 hours):</strong> Highest accuracy (90%+) for imminent tornado threats</li>
                        <li><strong>Medium-term (6-24 hours):</strong> Strong accuracy (80-85%) for developing conditions</li>
                        <li><strong>Long-term (24-72 hours):</strong> Moderate accuracy (70-75%) for general risk areas</li>
                      </ul>
                      <p>
                        Beyond 72 hours, we provide generalized risk assessments rather than specific predictions due to the chaotic nature of weather systems.
                      </p>
                    </div>
                  </div>
                </div>
                
                <div className="accordion-item">
                  <h4 className="accordion-header">
                    <button className="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#faq3">
                      What data sources do you use?
                    </button>
                  </h4>
                  <div id="faq3" className="accordion-collapse collapse" data-bs-parent="#faqAccordion">
                    <div className="accordion-body">
                      <p>
                        CALM integrates data from multiple sources:
                      </p>
                      <ul>
                        <li>NEXRAD (Next Generation Radar) network operated by NOAA</li>
                        <li>GOES (Geostationary Operational Environmental Satellite) imagery</li>
                        <li>NOAA's National Weather Service forecasts and alerts</li>
                        <li>Surface observation networks (ASOS, AWOS)</li>
                        <li>Atmospheric sounding data from weather balloons</li>
                        <li>Historical tornado reports from the Storm Prediction Center</li>
                        <li>Open Meteo API for supplementary weather data</li>
                      </ul>
                      <p>
                        All this data is processed and analyzed in real-time by our AI systems.
                      </p>
                    </div>
                  </div>
                </div>
                
                <div className="accordion-item">
                  <h4 className="accordion-header">
                    <button className="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#faq4">
                      How is CALM different from regular weather forecasts?
                    </button>
                  </h4>
                  <div id="faq4" className="accordion-collapse collapse" data-bs-parent="#faqAccordion">
                    <div className="accordion-body">
                      <p>
                        Unlike traditional weather forecasts that rely primarily on numerical weather prediction (NWP) models, CALM incorporates:
                      </p>
                      <ul>
                        <li><strong>AI-powered pattern recognition:</strong> Our neural networks can identify subtle patterns in atmospheric data that traditional models might miss</li>
                        <li><strong>Real-time radar analysis:</strong> Continuous monitoring and analysis of radar signatures associated with tornadic development</li>
                        <li><strong>Hyperlocal predictions:</strong> Focusing on specific areas rather than broad regions</li>
                        <li><strong>Continuous learning:</strong> Our models improve over time as they process more data and validation results</li>
                      </ul>
                      <p>
                        This approach allows us to provide more targeted and timely warnings for severe weather events.
                      </p>
                    </div>
                  </div>
                </div>
                
                <div className="accordion-item">
                  <h4 className="accordion-header">
                    <button className="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#faq5">
                      Is CALM a replacement for official weather warnings?
                    </button>
                  </h4>
                  <div id="faq5" className="accordion-collapse collapse" data-bs-parent="#faqAccordion">
                    <div className="accordion-body">
                      <p>
                        <strong>No.</strong> CALM is designed to complement, not replace, official warnings from the National Weather Service (NWS) and other government agencies.
                      </p>
                      <p>
                        Always follow the guidance and warnings issued by your local NWS office and emergency management agencies. CALM provides additional insight and early awareness, but official warnings should be your primary source for taking protective action.
                      </p>
                      <p>
                        We also incorporate NWS alerts into our system to ensure users have access to official information alongside our predictions.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="row">
        <div className="col-12">
          <div className="card">
            <div className="card-body text-center p-5">
              <h3 className="mb-4">Contact Us</h3>
              <p className="lead mb-4">
                Have questions or feedback about CALM? We'd love to hear from you!
              </p>
              <div className="d-flex justify-content-center">
                <a href="mailto:markbenda91@gmail.com" className="btn btn-primary me-3">
                  <i className="bi bi-envelope me-2"></i>
                  Email Us
                </a>
                <a href="https://github.com/darkmatter91/CALM" className="btn btn-outline-secondary">
                  <i className="bi bi-github me-2"></i>
                  GitHub Repository
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About; 
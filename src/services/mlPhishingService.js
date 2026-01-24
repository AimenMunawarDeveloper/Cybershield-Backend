/**
 * ML Phishing Detection Service
 * Integrates Python ML pipeline with Node.js backend
 */

const { spawn } = require('child_process');
const path = require('path');

class MLPhishingService {
  constructor() {
    this.modelPath = path.join(__dirname, '..', '..', 'models', 'phishing_detector.pth');
    this.scalerPath = path.join(__dirname, '..', '..', 'models', 'phishing_detector_scaler.pkl');
  }

  /**
   * Predict phishing probability for a user-reported incident
   * @param {Object} incidentData - Incident data from user report
   * @returns {Promise<Object>} Prediction results
   */
  async predictIncident(incidentData) {
    try {
      const result = await this._callPythonPredictor(incidentData);
      return result;
    } catch (error) {
      console.error('ML Prediction Error:', error);
      return {
        success: false,
        error: error.message,
        is_phishing: null,
        phishing_probability: null,
        confidence: null
      };
    }
  }

  /**
   * Call Python ML pipeline for prediction
   * @private
   */
  _callPythonPredictor(incidentData) {
    return new Promise((resolve, reject) => {
      const pythonCode = `
import sys
import json
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'ml_pipeline'))

from inference_service import PhishingInferenceService

try:
    # Read incident data from stdin
    incident_json = sys.stdin.read()
    incident_data = json.loads(incident_json)
    
    # Initialize service
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'phishing_detector.pth')
    scaler_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'phishing_detector_scaler.pkl')
    
    service = PhishingInferenceService(model_path=model_path, scaler_path=scaler_path)
    
    # Predict
    result = service.predict_incident(incident_data)
    
    # Output result as JSON
    print(json.dumps(result))
except Exception as e:
    error_result = {
        'success': False,
        'error': str(e),
        'is_phishing': None
    }
    print(json.dumps(error_result))
    sys.exit(1)
      `;

      const python = spawn('python', ['-c', pythonCode], {
        cwd: path.join(__dirname, '..', '..')
      });

      let stdout = '';
      let stderr = '';

      python.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      python.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      python.on('close', (code) => {
        if (code !== 0) {
          reject(new Error(`Python process exited with code ${code}: ${stderr}`));
          return;
        }

        try {
          const result = JSON.parse(stdout.trim());
          resolve(result);
        } catch (parseError) {
          reject(new Error(`Failed to parse Python output: ${stdout}`));
        }
      });

      python.on('error', (error) => {
        reject(new Error(`Failed to spawn Python process: ${error.message}`));
      });

      // Send incident data to Python
      python.stdin.write(JSON.stringify(incidentData));
      python.stdin.end();
    });
  }

  /**
   * Extract features from incident (for analysis/debugging)
   * @param {Object} incidentData - Incident data
   * @returns {Promise<Object>} Extracted features
   */
  async extractFeatures(incidentData) {
    try {
      const pythonCode = `
import sys
import json
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'ml_pipeline'))

from inference_service import PhishingInferenceService

try:
    incident_json = sys.stdin.read()
    incident_data = json.loads(incident_json)
    
    service = PhishingInferenceService()
    result = service.extract_features(incident_data)
    
    print(json.dumps(result))
except Exception as e:
    error_result = {
        'success': False,
        'error': str(e)
    }
    print(json.dumps(error_result))
    sys.exit(1)
      `;

      const python = spawn('python', ['-c', pythonCode], {
        cwd: path.join(__dirname, '..', '..')
      });

      let stdout = '';
      let stderr = '';

      python.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      python.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      return new Promise((resolve, reject) => {
        python.on('close', (code) => {
          if (code !== 0) {
            reject(new Error(`Python process exited with code ${code}: ${stderr}`));
            return;
          }

          try {
            const result = JSON.parse(stdout.trim());
            resolve(result);
          } catch (parseError) {
            reject(new Error(`Failed to parse Python output: ${stdout}`));
          }
        });

        python.on('error', (error) => {
          reject(new Error(`Failed to spawn Python process: ${error.message}`));
        });

        python.stdin.write(JSON.stringify(incidentData));
        python.stdin.end();
      });
    } catch (error) {
      console.error('Feature Extraction Error:', error);
      return {
        success: false,
        error: error.message,
        features: null
      };
    }
  }

  /**
   * Format incident for ML pipeline (phishing vs non-phishing from message content only).
   * @param {Object} reportData - Raw incident: message/text, messageType, metadata, urls, etc.
   * @returns {Object} Formatted incident for prediction
   */
  formatIncidentForML(reportData) {
    return {
      text: reportData.message || reportData.text || '',
      message_type: reportData.messageType || reportData.message_type || 'email',
      metadata: {
        from: reportData.from || reportData.sender || '',
        from_email: reportData.fromEmail || reportData.sender_email || '',
        subject: reportData.subject || '',
        date: reportData.date || reportData.timestamp || new Date().toISOString(),
        to: reportData.to || [],
        cc: reportData.cc || [],
        bcc: reportData.bcc || [],
        headers: reportData.headers || {},
        ...(reportData.metadata || {})
      },
      urls: reportData.urls || reportData.links || [],
      html_content: reportData.htmlContent || reportData.html_content || null
    };
  }
}

module.exports = new MLPhishingService();

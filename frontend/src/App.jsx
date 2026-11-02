
import { useEffect, useState } from "react";
import Papa from "papaparse";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [apiStatus, setApiStatus] = useState("Checking...");
  const [selectedFile, setSelectedFile] = useState(null);
  const [features, setFeatures] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then((response) => {
        if (!response.ok) throw new Error("API unavailable");
        return response.json();
      })
      .then(() => setApiStatus("Online"))
      .catch(() => setApiStatus("Offline"));
  }, []);

  const handleFileChange = (event) => {
    const file = event.target.files[0];

    if (!file) return;

    setSelectedFile(file);
    setFeatures(null);
    setResult(null);
    setError("");

    Papa.parse(file, {
      header: true,
      preview: 1,
      skipEmptyLines: true,

      transformHeader: (header) => header.trim(),

      complete: (parsed) => {
        if (!parsed.data || !parsed.data.length) {
          setError("No data found in the CSV file.");
          return;
        }

        const row = parsed.data[0];
        const numericFeatures = {};

        Object.entries(row).forEach(([key, value]) => {
          if (key === "Label" || key === "Target" || key === "") return;

          const number = Number(value);

          if (value !== "" && Number.isFinite(number)) {
            numericFeatures[key] = number;
          }
        });

        if (Object.keys(numericFeatures).length === 0) {
          setError("No valid numeric features found.");
          return;
        }

        setFeatures(numericFeatures);
      },

      error: () => {
        setError("Unable to read the CSV file.");
      }
    });
  };

  const analyzeTraffic = async () => {
    if (!features) {
      setError("Please upload a valid CSV file first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          features: features
        })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Unable to connect to the backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <aside className="sidebar">
        <h2>🛡 CyberGuard<span>-AI</span></h2>

        <nav>
          <a className="active" href="#dashboard">Dashboard</a>
          <a href="#detection">Threat Detection</a>
          <a href="#about">About Project</a>
        </nav>

        <div className="sidebar-bottom">
          <p>Machine Learning</p>
          <small>Network Intrusion Detection</small>
        </div>
      </aside>

      <main className="main-content" id="dashboard">
        <header className="topbar">
          <div>
            <h1>Network Security Dashboard</h1>
            <p>Intelligent Network Intrusion Detection System</p>
          </div>

          <div className="api-status">
            <span className={apiStatus === "Online" ? "dot online" : "dot offline"} />
            API: {apiStatus}
          </div>
        </header>

        <section className="welcome">
          <div>
            <h2>Welcome to CyberGuard-AI</h2>
            <p>
              Detect malicious network traffic using Machine Learning
              and XGBoost classification.
            </p>
          </div>
          <div className="shield">🛡️</div>
        </section>

        <section className="stats">
          <div className="stat-card">
            <span>ML Algorithm</span>
            <h2>XGBoost</h2>
            <p>Trained classification model</p>
          </div>

          <div className="stat-card">
            <span>Input Features</span>
            <h2>78</h2>
            <p>Network traffic attributes</p>
          </div>

          <div className="stat-card">
            <span>Backend Status</span>
            <h2>{apiStatus}</h2>
            <p>FastAPI service</p>
          </div>
        </section>

        <section className="detection-section" id="detection">
          <div className="section-heading">
            <h2>Network Traffic Analysis</h2>
            <p>Upload a CIC-IDS2017 CSV file to analyze network traffic.</p>
          </div>

          <div className="upload-box">
            <div className="upload-icon">📂</div>
            <h3>Upload Network Traffic CSV</h3>
            <p>Select a cleaned CIC-IDS2017 CSV dataset.</p>

            <label className="upload-button">
              Choose CSV File
              <input
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                hidden
              />
            </label>

            {selectedFile && (
              <p className="file-name">
                Selected: {selectedFile.name}
              </p>
            )}

            {features && (
              <p className="success-text">
                ✓ {Object.keys(features).length} numeric features loaded
              </p>
            )}

            <button
              className="analyze-button"
              onClick={analyzeTraffic}
              disabled={!features || loading || apiStatus !== "Online"}
            >
              {loading ? "Analyzing..." : "Analyze Network Traffic"}
            </button>

            <p className="note">
              Note: Only the first data row of the uploaded CSV is analyzed.
            </p>
          </div>
        </section>

        {error && (
          <div className="error-message">
            ⚠ {error}
          </div>
        )}

        {result && (
          <section className="result-section">
            <h2>Threat Detection Result</h2>

            <div className={`result-card ${result.target === 1 ? "attack" : "benign"}`}>
              <h3>{result.prediction}</h3>
              <p>{result.message}</p>

              <div className="result-details">
                <div>
                  <span>Prediction</span>
                  <strong>{result.target}</strong>
                </div>

                <div>
                  <span>Confidence</span>
                  <strong>{result.confidence}%</strong>
                </div>

                <div>
                  <span>Risk Level</span>
                  <strong>{result.risk_level}</strong>
                </div>
              </div>
            </div>
          </section>
        )}

        <section className="about-section" id="about">
          <h2>About CyberGuard-AI</h2>
          <p>
            CyberGuard-AI is an intelligent network intrusion detection
            prototype developed using Machine Learning and the CIC-IDS2017
            dataset. It uses an XGBoost model to classify network traffic
            as benign or potentially malicious.
          </p>
        </section>

        <footer>
          CyberGuard-AI | M.Tech Computer Science Project | 2026
        </footer>
      </main>
    </div>
  );
}

export default App;

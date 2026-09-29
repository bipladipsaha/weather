const { execFile } = require('child_process');
const path = require('path');

class AIPipelineAdapter {
    constructor() {
        this.pythonPath = 'python'; // Assuming python is in PATH
        // Go up one level from services, then up to the root to find ai/pipeline/e2e_runner.py
        this.runnerScript = path.resolve(__dirname, '../../ai/pipeline/e2e_runner.py');
    }

    /**
     * Helper to run the python script and parse JSON
     */
    _runScript(args) {
        return new Promise((resolve, reject) => {
            execFile(this.pythonPath, [this.runnerScript, ...args], (error, stdout, stderr) => {
                if (error) {
                    console.error("AI Pipeline Error:", stderr);
                    return reject(error);
                }
                try {
                    const data = JSON.parse(stdout);
                    resolve(data);
                } catch (parseError) {
                    console.error("Failed to parse AI Pipeline output:", stdout);
                    reject(parseError);
                }
            });
        });
    }

    async getEvents(physicsMode = "PASS") {
        return [{
            "id": "EVT-MOCK1234",
            "hazard_type": "heat",
            "category": "heat",
            "type": "Heat Anomaly",
            "severityScore": 90,
            "probability": 95,
            "area": 1.0,
            "currentLead": "T+24h",
            "location": "Mock Location",
            "lat": 23.0,
            "lng": 80.0
        }];
    }

    async getHealth() {
        return { status: "ok" };
    }
}

module.exports = new AIPipelineAdapter();

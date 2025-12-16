// Use relative paths; Vercel handles /api routes automatically
const API_BASE = "/api";

async function loadStats() {
    const outputEl = document.getElementById('stats-output');
    try {
        outputEl.textContent = 'Loading...';
        const response = await fetch(`${API_BASE}/stats`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        outputEl.textContent = JSON.stringify(data, null, 2);
    } catch (error) {
        outputEl.textContent = `Error: ${error.message}`;
    }
}

async function askQuestion() {
    const inputEl = document.getElementById('json-input');
    const outputEl = document.getElementById('json-output');
    try {
        const inputJson = JSON.parse(inputEl.value);
        if (!inputJson.question) {
            alert('Input JSON must contain a "question" field');
            return;
        }

        const response = await fetch(`${API_BASE}/prompt`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(inputJson)
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        outputEl.textContent = JSON.stringify(data, null, 2);
    } catch (error) {
        outputEl.textContent = `Error: ${error.message}`;
    }
}

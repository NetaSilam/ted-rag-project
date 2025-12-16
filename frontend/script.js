// Load stats on page load
document.addEventListener('DOMContentLoaded', () => {
    // Don't auto-load, wait for user to click GET button
});

async function loadStats() {
    const outputEl = document.getElementById('stats-output');

    try {
        outputEl.textContent = 'Loading...';

        const response = await fetch('https://ted-rag-project-production.up.railway.app/api/stats');

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();

        // Display formatted JSON
        outputEl.textContent = JSON.stringify(data, null, 2);

    } catch (error) {
        console.error('Stats error:', error);
        outputEl.textContent = `Error: ${error.message}`;
    }
}

async function askQuestion() {
    const inputEl = document.getElementById('json-input');
    const outputEl = document.getElementById('json-output');
    const btn = document.querySelector('.post-btn');
    const btnText = document.getElementById('btn-text');
    const spinner = document.getElementById('spinner');

    try {
        // Parse input JSON
        const inputJson = JSON.parse(inputEl.value);

        if (!inputJson.question) {
            alert('Input JSON must contain a "question" field');
            return;
        }

        // Show loading state
        btn.disabled = true;
        btnText.style.display = 'none';
        spinner.style.display = 'inline-block';
        outputEl.textContent = 'Sending request...';

        // Make request
        const response = await fetch('https://ted-rag-project-production.up.railway.app/api/prompt', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(inputJson)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Display formatted JSON output
        outputEl.textContent = JSON.stringify(data, null, 2);

    } catch (error) {
        if (error instanceof SyntaxError) {
            alert('Invalid JSON input. Please check your JSON syntax.');
            outputEl.textContent = 'Error: Invalid JSON input';
        } else {
            alert('Error: ' + error.message);
            outputEl.textContent = `Error: ${error.message}`;
        }
        console.error('Error:', error);
    } finally {
        // Reset button state
        btn.disabled = false;
        btnText.style.display = 'inline';
        spinner.style.display = 'none';
    }
}
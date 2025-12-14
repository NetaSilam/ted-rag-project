// Load stats on page load
document.addEventListener('DOMContentLoaded', loadStats);

async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        document.getElementById('stats').innerHTML = `
            Chunk Size: ${data.chunk_size} | 
            Overlap Ratio: ${data.overlap_ratio} | 
            Top K: ${data.top_k}
        `;
    } catch (error) {
        document.getElementById('stats').innerHTML = '❌ Failed to load stats';
        console.error('Stats error:', error);
    }
}

async function askQuestion() {
    const question = document.getElementById('question').value.trim();

    if (!question) {
        alert('Please enter a question!');
        return;
    }

    // Show loading state
    const btn = document.getElementById('submit-btn');
    const btnText = document.getElementById('btn-text');
    const spinner = document.getElementById('spinner');

    btn.disabled = true;
    btnText.style.display = 'none';
    spinner.style.display = 'inline-block';

    try {
        const response = await fetch('/api/prompt', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: question })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Error:', error);
    } finally {
        // Reset button state
        btn.disabled = false;
        btnText.style.display = 'inline';
        spinner.style.display = 'none';
    }
}

function displayResults(data) {
    // Show results section
    document.getElementById('results').style.display = 'block';

    // Display response
    document.getElementById('response').textContent = data.response;

    // Display context
    const contextHtml = data.context.map((ctx, index) => `
        <div class="context-item">
            <strong>Context ${index + 1} (Score: ${ctx.score.toFixed(4)})</strong>
            <div class="context-meta">
                Talk ID: ${ctx.talk_id} | Title: ${ctx.title}
            </div>
            <div class="context-chunk">
                ${ctx.chunk}
            </div>
        </div>
    `).join('');
    document.getElementById('context').innerHTML = contextHtml;

    // Display prompts
    document.getElementById('system-prompt').textContent = data.Augmented_prompt.System;
    document.getElementById('user-prompt').textContent = data.Augmented_prompt.User;

    // Scroll to results
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
}

// Allow Enter key to submit (Shift+Enter for new line)
document.getElementById('question').addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        askQuestion();
    }
});
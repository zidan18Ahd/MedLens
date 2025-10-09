const API_BASE = 'http://localhost:8000/api';

// Upload file function
async function uploadFile(type) {
    const fileInput = type === 'prescription' ? 
        document.getElementById('prescriptionUpload') : 
        document.getElementById('pdfUpload');
    
    if (!fileInput.files[0]) {
        alert('Please select a file first');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('doc_type', type);

    try {
        showLoading(true);
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        
        if (result.status === 'success') {
            alert(`File processed successfully! Found ${result.result.entities_found} entities.`);
            fileInput.value = ''; // Clear input
        } else {
            alert('Error processing file: ' + result.detail);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// Upload text function
async function uploadText() {
    const textContent = document.getElementById('textContent').value.trim();
    
    if (!textContent) {
        alert('Please enter some text');
        return;
    }

    const formData = new FormData();
    formData.append('file', new Blob([textContent], { type: 'text/plain' }), 'text.txt');
    formData.append('doc_type', 'text');

    try {
        showLoading(true);
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        
        if (result.status === 'success') {
            alert(`Text processed successfully! Found ${result.result.entities_found} entities.`);
            document.getElementById('textContent').value = '';
        } else {
            alert('Error processing text: ' + result.detail);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// Web scraping function
async function scrapeWeb() {
    const url = document.getElementById('webUrl').value.trim();
    
    if (!url) {
        alert('Please enter a URL');
        return;
    }

    const formData = new FormData();
    formData.append('url', url);

    try {
        showLoading(true);
        const response = await fetch(`${API_BASE}/webscrape`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        
        if (result.status === 'success') {
            alert('Web content added successfully!');
            document.getElementById('webUrl').value = '';
        } else {
            alert('Error scraping website: ' + result.detail);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// Ask question function
async function askQuestion() {
    const question = document.getElementById('questionInput').value.trim();
    const mode = document.getElementById('modeSelect').value;
    
    if (!question) {
        alert('Please enter a question');
        return;
    }

    const formData = new FormData();
    formData.append('question', question);
    formData.append('mode', mode);
    formData.append('top_k', 5);

    try {
        showLoading(true);
        const response = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        
        if (result.status === 'success') {
            displayResults(result.result);
        } else {
            alert('Error querying system: ' + result.detail);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// Display results function
function displayResults(result) {
    // Display answer
    const answerCard = document.getElementById('answerCard');
    const answerContent = document.getElementById('answerContent');
    
    answerContent.textContent = result.answer;
    answerCard.style.display = 'block';

    // Display context
    const contextCard = document.getElementById('contextCard');
    const contextContent = document.getElementById('contextContent');
    
    if (result.context && result.context.length > 0) {
        contextContent.innerHTML = result.context
            .map(ctx => `<div class="context-item">${ctx}</div>`)
            .join('');
        contextCard.style.display = 'block';
    } else {
        contextCard.style.display = 'none';
    }

    // Display entities
    const entitiesCard = document.getElementById('entitiesCard');
    const entitiesContent = document.getElementById('entitiesContent');
    
    if (result.entities && result.entities.length > 0) {
        entitiesContent.innerHTML = result.entities
            .map(entity => `
                <span class="entity-tag" title="${entity.description}">
                    ${entity.name} (${entity.type})
                </span>
            `)
            .join('');
        entitiesCard.style.display = 'block';
    } else {
        entitiesCard.style.display = 'none';
    }
}

// Utility function to show loading state
function showLoading(loading) {
    const buttons = document.querySelectorAll('button');
    buttons.forEach(button => {
        if (loading) {
            button.classList.add('loading');
            button.disabled = true;
        } else {
            button.classList.remove('loading');
            button.disabled = false;
        }
    });
}

// Health check on page load
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const result = await response.json();
        console.log('System health:', result);
    } catch (error) {
        console.log('Backend not reachable. Please start the server.');
    }
});
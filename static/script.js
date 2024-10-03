document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleUpload);
    }

    const deleteButtons = document.querySelectorAll('.delete-btn');
    deleteButtons.forEach(button => {
        button.addEventListener('click', handleDelete);
    });

    const feedbackForm = document.getElementById('feedback-form');
    if (feedbackForm) {
        feedbackForm.addEventListener('submit', handleFeedbackSubmission);
    }

    // API Usage modal
    const showApiUsageButton = document.getElementById('show-api-usage');
    const apiUsageModal = document.getElementById('api-usage-modal');
    const closeButton = apiUsageModal.querySelector('.close');

    showApiUsageButton.addEventListener('click', function() {
        fetchApiUsage();
        apiUsageModal.style.display = 'block';
    });

    closeButton.addEventListener('click', function() {
        apiUsageModal.style.display = 'none';
    });

    window.addEventListener('click', function(event) {
        if (event.target === apiUsageModal) {
            apiUsageModal.style.display = 'none';
        }
    });

    // Start polling for document status
    pollDocumentStatus();

    // Add event listeners for export buttons
    const exportButtons = document.querySelectorAll('.export-buttons a');
    exportButtons.forEach(button => {
        button.addEventListener('click', handleExport);
    });
});

async function handleUpload(e) {
    e.preventDefault();
    const form = e.target;
    const formData = new FormData(form);
    
    const loadingElement = document.getElementById('loading');
    const messageElement = document.getElementById('message');
    const errorMessageElement = document.getElementById('error-message');
    
    if (loadingElement) loadingElement.classList.remove('hidden');
    if (messageElement) messageElement.classList.add('hidden');
    if (errorMessageElement) errorMessageElement.classList.add('hidden');
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            if (messageElement) {
                messageElement.textContent = result.message;
                messageElement.classList.remove('hidden');
            }
            pollDocumentStatus();
        } else {
            throw new Error(result.error || 'An error occurred');
        }
    } catch (error) {
        handleError(error);
    } finally {
        if (loadingElement) loadingElement.classList.add('hidden');
    }
}

async function pollDocumentStatus() {
    const pollInterval = 5000;
    const maxAttempts = 120;
    let attempts = 0;

    const poll = async () => {
        try {
            const response = await fetch('/get_checklists');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const result = await response.json();

            displayDocumentStatus(result);

            if (result.completed < result.total_documents) {
                attempts++;
                if (attempts < maxAttempts) {
                    setTimeout(poll, pollInterval);
                } else {
                    const messageElement = document.getElementById('message');
                    if (messageElement) {
                        messageElement.textContent = 'Document processing is taking longer than expected. Please check back later.';
                    }
                }
            }
        } catch (error) {
            handleError(error);
        }
    };

    poll();
}

function displayDocumentStatus(result) {
    const statusContainer = document.getElementById('document-status');
    if (!statusContainer) return;

    statusContainer.innerHTML = '';

    // Display processing documents
    for (const [filename, status] of Object.entries(result.processing)) {
        const processingElement = createStatusElement(filename, 'processing', status);
        statusContainer.appendChild(processingElement);
    }

    // Display completed documents
    for (const [filename, checklist] of Object.entries(result.checklists)) {
        const completedElement = createStatusElement(filename, 'completed', checklist);
        statusContainer.appendChild(completedElement);
    }

    // Display failed documents
    for (const [filename, error] of Object.entries(result.errors)) {
        const failedElement = createStatusElement(filename, 'failed', error);
        statusContainer.appendChild(failedElement);
    }

    statusContainer.classList.remove('hidden');
}

function createStatusElement(filename, status, content) {
    const element = document.createElement('div');
    element.className = `status-item ${status}`;
    
    const header = document.createElement('h3');
    header.textContent = filename;
    element.appendChild(header);

    const statusText = document.createElement('p');
    if (status === 'processing') {
        statusText.textContent = content; // This now includes the attempt number
    } else {
        statusText.textContent = status === 'completed' ? 'Completed' : 'Failed';
    }
    element.appendChild(statusText);

    if (status === 'processing') {
        const spinner = document.createElement('div');
        spinner.className = 'spinner';
        element.appendChild(spinner);
    } else if (status === 'completed') {
        const viewButton = document.createElement('button');
        viewButton.textContent = 'View Checklist';
        viewButton.addEventListener('click', () => viewChecklist(filename, content));
        element.appendChild(viewButton);
    } else if (status === 'failed') {
        const errorText = document.createElement('p');
        errorText.textContent = content;
        element.appendChild(errorText);

        const retryButton = document.createElement('button');
        retryButton.textContent = 'Retry';
        retryButton.addEventListener('click', () => retryProcessing(filename));
        element.appendChild(retryButton);
    }

    return element;
}

function viewChecklist(filename, content) {
    const checklistContainer = document.getElementById('checklist-container');
    if (!checklistContainer) return;

    checklistContainer.innerHTML = `
        <h2>${filename} Checklist</h2>
        <ul>
            ${content.map((item, index) => `
                <li>
                    <span class="item-content">${item[0]}</span>
                    <span class="item-score">${(item[1] * 100).toFixed(2)}%</span>
                    <button class="explain-btn" onclick="explainItem('${filename}', ${index})">Explain</button>
                </li>
            `).join('')}
        </ul>
    `;
    checklistContainer.classList.remove('hidden');
}

async function explainItem(filename, itemIndex) {
    try {
        const response = await fetch(`/explain/${filename}/${itemIndex}`);
        const data = await response.json();
        
        if (response.ok) {
            displayExplanation(data);
        } else {
            throw new Error(data.error || 'An error occurred while fetching the explanation');
        }
    } catch (error) {
        handleError(error);
    }
}

function displayExplanation(data) {
    const modal = document.getElementById('explanation-modal');
    const content = document.getElementById('explanation-content');
    
    content.innerHTML = `
        <h3>Checklist Item:</h3>
        <p>${data.item}</p>
        <h3>Confidence Score: ${(data.score * 100).toFixed(2)}%</h3>
        <h3>Explanation:</h3>
        <div>
            ${data.explanation.map(token => `
                <span class="token" style="background-color: rgba(52, 152, 219, ${Math.abs(token.attribution)});">
                    ${token.token}
                </span>
            `).join('')}
        </div>
    `;
    
    modal.style.display = 'block';
    
    const closeButton = modal.querySelector('.close');
    closeButton.onclick = function() {
        modal.style.display = 'none';
    }
    
    window.onclick = function(event) {
        if (event.target == modal) {
            modal.style.display = 'none';
        }
    }
}

async function retryProcessing(filename) {
    try {
        const response = await fetch(`/retry/${filename}`, { method: 'POST' });
        const result = await response.json();
        if (response.ok) {
            alert(`Retrying processing for ${filename}`);
            pollDocumentStatus();
        } else {
            throw new Error(result.error || 'An error occurred');
        }
    } catch (error) {
        handleError(error);
    }
}

function handleDelete(e) {
    e.preventDefault();
    const filename = e.target.dataset.filename;
    if (filename && confirm(`Are you sure you want to delete ${filename}?`)) {
        e.target.closest('form').submit();
    }
}

async function handleFeedbackSubmission(e) {
    e.preventDefault();
    const form = e.target;
    const formData = new FormData(form);
    const filename = form.action.split('/').pop();

    try {
        const response = await fetch(form.action, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(Object.fromEntries(formData)),
        });

        const result = await response.json();

        if (response.ok) {
            alert('Feedback submitted successfully');
            form.reset();
        } else {
            throw new Error(result.error || 'An error occurred');
        }
    } catch (error) {
        handleError(error);
    }
}

async function fetchApiUsage() {
    try {
        const response = await fetch('/api_usage');
        const data = await response.json();
        displayApiUsage(data);
    } catch (error) {
        console.error('Error fetching API usage:', error);
        document.getElementById('api-usage-content').innerHTML = '<p>Error fetching API usage data</p>';
    }
}

function displayApiUsage(data) {
    const container = document.getElementById('api-usage-content');
    let html = '<table><tr><th>API Name</th><th>Usage Count</th></tr>';
    for (const [apiName, count] of Object.entries(data)) {
        html += `<tr><td>${apiName}</td><td>${count}</td></tr>`;
    }
    html += '</table>';
    container.innerHTML = html;
}

function handleError(error) {
    console.error('Error:', error);
    let errorMessage = error.message;
    if (error.message.includes('rate limit')) {
        errorMessage = 'API rate limit exceeded. Please try again later.';
    }
    const errorMessageElement = document.getElementById('error-message');
    if (errorMessageElement) {
        errorMessageElement.textContent = errorMessage;
        errorMessageElement.classList.remove('hidden');
    }
}

function handleExport(e) {
    e.preventDefault();
    const url = e.target.href;
    const format = url.includes('csv') ? 'CSV' : 'Excel';
    
    fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.blob();
        })
        .then(blob => {
            const downloadUrl = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = downloadUrl;
            a.download = `checklist.${format.toLowerCase()}`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(downloadUrl);
        })
        .catch(error => {
            console.error('Error exporting checklist:', error);
            alert(`Error exporting checklist as ${format}. Please try again.`);
        });
}

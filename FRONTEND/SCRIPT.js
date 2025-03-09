function previewImage() {
    const fileInput = document.getElementById("file-input");
    const previewContainer = document.getElementById("preview-container");
    const previewImage = document.getElementById("preview-image");
    const uploadBtn = document.getElementById("upload-btn");
    const file = fileInput.files[0];

    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            previewImage.src = e.target.result;
            previewContainer.style.display = "block";
        };
        reader.readAsDataURL(file);
        uploadBtn.disabled = false;
    }
}

document.getElementById('upload-form').addEventListener('submit', function (e) {
    e.preventDefault(); // Prevent the default form submission

    const fileInput = document.getElementById('file-input');
    const file = fileInput.files[0];
    const statusMessage = document.getElementById('status-message');
    const resultContainer = document.getElementById('result-container');

    if (!file) {
        alert("Please select a file before submitting.");
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    statusMessage.textContent = "Uploading and processing...";
    resultContainer.style.display = "none";

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            alert(data.error);
            return;
        }

        // Display the prediction and confidence score
        document.getElementById('prediction').innerText = `Prediction: ${data.predicted_label}`;
        document.getElementById('confidence').innerText = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;

        // Display the original image
        const originalImage = document.getElementById('original-image');
        originalImage.src = `data:image/jpeg;base64,${data.original_image}`;

        // Display the heatmap
        const heatmapImage = document.getElementById('heatmap-image');
        heatmapImage.src = `data:image/jpeg;base64,${data.heatmap}`;

        // Show the result container
        resultContainer.style.display = "block";
        statusMessage.textContent = "Prediction completed successfully!";
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while processing the request. Please try again.');
        statusMessage.textContent = "An error occurred. Please try again.";
    });
});
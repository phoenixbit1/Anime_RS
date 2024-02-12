document.addEventListener('DOMContentLoaded', function() { // Wait for the DOM to load
    const form = document.getElementById('recommendation-form'); // Get the form element
    const recommendationsContainer = document.getElementById('recommendations-container'); // Get the recommendations container element

    form.addEventListener('submit', function(e) {
        e.preventDefault(); // Prevent the default form submission

        const title = document.getElementById('anime-title').value;
        const numberOfRecs = document.getElementById('num-recs').value;

        fetch('/recommendations', {
            method: 'POST', // Send a POST request
            headers: { // Set the Content-Type header to application/json
                'Content-Type': 'application/json' 
            },
            body: JSON.stringify({ title: title, n: numberOfRecs }) // The data to send
        })
        .then(response => response.json()) // Parse the JSON response
        .then(data => {
            // Clear previous recommendations
            recommendationsContainer.innerHTML = '';

            // Display the recommendations on the page
            data.model1.forEach(function(rec) {
                const recElement = document.createElement('p'); // Create a new paragraph element
                recElement.textContent = rec; // Set the text of the paragraph to the recommendation
                recommendationsContainer.appendChild(recElement); // Add the paragraph to the recommendations container
            });
        })
        .catch(error => console.error('Error:', error)); // Log any errors that occur
    });
});
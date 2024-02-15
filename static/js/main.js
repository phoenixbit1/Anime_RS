document.addEventListener('DOMContentLoaded', function() { // When the document is ready
    const form = document.getElementById('recommendation-form'); // Get the recommendation form
    const recommendationsContainer = document.getElementById('recommendations-container'); // Get the recommendations container
    const loader = document.getElementById('loader'); // Get the loader element

    // Fetch anime titles for autocomplete when the document is ready
    $.getJSON('/anime-titles', function(data) {
        $('#anime-title').autocomplete({
            source: data.titles,
            minLength: 2
        });
    });

    form.addEventListener('submit', function(e) {
        e.preventDefault(); // Prevent the default form submission

        const title = document.getElementById('anime-title').value;
        const numberOfRecs = document.getElementById('num-recs').value;
        
        // Show loader
        loader.style.display = 'block';

        fetch('/recommendations', { // Send a POST request to the server
            method: 'POST', // Use the POST method
            headers: { // Set the request headers
                'Content-Type': 'application/json' // Send the request body in JSON format
            },
            body: JSON.stringify({ title: title, n: numberOfRecs }) // Set the request body
        })
        .then(response => { // Parse the response
            if (!response.ok) { // If the response status is not OK (200)
                throw new Error(`HTTP error! status: ${response.status}`); // Throw an error
            }
            return response.json(); // Parse the response body as JSON
        }) 
        .then(data => {
            recommendationsContainer.innerHTML = ''; // Clear the recommendations container
            // Hide loader
            loader.style.display = 'none';

            // Loop over each model's data and populate the respective container
            Object.keys(data).forEach(function(modelKey) {
                console.log(`Processing ${modelKey}`); // Log the current model being processed
                const modelContainer = document.getElementById(`recommendations-${modelKey}`);
                if (!modelContainer) { // If the container is not found
                    console.error(`Container not found for model: ${modelKey}`); // Log if container is not found
                    return; // Skip to the next model
                }
                modelContainer.innerHTML = `<h3>Model ${modelKey.replace('model', '')} Recommendations:</h3>`; // Set the container's inner HTML
                const modelRecommendations = data[modelKey]; // Get the recommendations for the current model

                
                // Check if recommendations are an array of strings or objects with title and explanation
                if (Array.isArray(modelRecommendations) && modelRecommendations.length > 0 && typeof modelRecommendations[0] === 'string') {
                    modelRecommendations.forEach(function(rec) { // Loop over each recommendation
                        const recElement = document.createElement('p'); // create a new element for the recommendation
                        recElement.textContent = rec; // Set the element's text content to the recommendation
                        modelContainer.appendChild(recElement); // Append the element to the model's container
                    });
                } 
                
                else if (Array.isArray(modelRecommendations[0])) { // Check if recommendations are an array of arrays
                    modelRecommendations.forEach(function(rec) { // Loop over each recommendation
                        const recExplanation = document.createElement('p'); // create a new element for the recommendation
                        recExplanation.innerHTML = `<strong>${rec[0]}</strong>: ${rec[1]}`; // Set the element's text content to the recommendation
                        modelContainer.appendChild(recExplanation); // Append the element to the model's container
                    });
                }
                 else {
                    console.error(`Unexpected data format for model: ${modelKey}`); // Log if data format is unexpected
                }
            });
        })
        .catch(error => { // Catch any errors
            console.error('Error:', error); // Log the error
            loader.style.display = 'none'; // Hide loader
        });
    });
});

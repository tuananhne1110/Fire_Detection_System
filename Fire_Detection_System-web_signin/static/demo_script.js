document.addEventListener("DOMContentLoaded", function () {
    const dropZone = document.getElementById("dropZone");
    const originalImage = document.getElementById("originalImage");
    const originalImageContainer = document.getElementById("originalImageContainer");
    const fileInput = document.getElementById("fileInput");

    // Prevent default behavior for drop events
    dropZone.addEventListener("dragover", function (e) {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", function () {
        dropZone.classList.remove("drag-over");
    });

    dropZone.addEventListener("drop", function (e) {
        e.preventDefault();
        dropZone.classList.remove("drag-over");

        const file = e.dataTransfer.files[0];
        handleImage(file);
    });

    // Click event on drop zone triggers the file input
    dropZone.addEventListener("click", function () {
        fileInput.click();
    });

    // Input file change event
    fileInput.addEventListener("change", function () {
        const file = fileInput.files[0];
        handleImage(file);
    });
    // scrol down
    // JavaScript code
    function handleImage(file) {
        if (file) {
            const fileSizeInBytes = file.size
            
            const reader = new FileReader();

            reader.onload = function (e) {
                originalImage.src = e.target.result;

                // Show image containers once an image is loaded
                originalImageContainer.style.display = "flex";
                originalImageContainer.style.flexDirection = "row";
                originalImageContainer.style.justifyContent = "space-between";

                originalImage.scrollIntoView({
                    behavior: 'smooth',
                    block: 'center',
                    inline: 'center' 
                });

            };
            // Create a FormData object to send the image to the server
            const formData = new FormData();
            formData.append('image_blob', file);

            // Send the image for processing to the server
            fetch('/process-image', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Display the grayscale image
                if (data.gray_image) {
                    const grayscaleImage = document.getElementById('grayscaleImage');
                    grayscaleImage.src = 'data:image/jpeg;base64,' + data.gray_image;
                    grayscaleImage.classList.add('grayscale-image');
                }
            })
            .catch(error => {
                console.error('Error processing the image:', error);
            });
            reader.readAsDataURL(file);
        }
    }
    // function handleImageUpload() {
    //     const fileInput = document.getElementById("fileInput");
    //     const file = fileInput.files[0]; // Get the first selected file (assuming only one file is selected)
    
    //     if (file) {
    //         const fileSizeInBytes = file.size;
    //         const fileSizeInKilobytes = fileSizeInBytes / 1024;
    //         const fileSizeInMegabytes = fileSizeInKilobytes / 1024;
    
    //         console.log("File size in bytes: " + fileSizeInBytes);
    //         console.log("File size in kilobytes: " + fileSizeInKilobytes);
    //         console.log("File size in megabytes: " + fileSizeInMegabytes);
    //     }
    // }
});

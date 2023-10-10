document.addEventListener("DOMContentLoaded", function () {
    const dropZone = document.getElementById("dropZone");
    const originalImage = document.getElementById("originalImage");
    // const grayscaleImage = document.getElementById("grayscaleImage");
    const originalImageContainer = document.getElementById("originalImageContainer");
    // const grayscaleImageContainer = document.getElementById("grayscaleImageContainer");
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
            const reader = new FileReader();

            reader.onload = function (e) {
                originalImage.src = e.target.result;

                // Show image containers once an image is loaded
                originalImageContainer.style.display = "block";

                // Scroll to the end of the page with smooth behavior
                window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
            };

            reader.readAsDataURL(file);
        }
    }
});

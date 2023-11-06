document.addEventListener("DOMContentLoaded", function () {
    const dropZone = document.getElementById("dropZone");
    const originalImage = document.getElementById("originalImage");
    const originalImageContainer = document.getElementById("originalImageContainer");
    const fileInput = document.getElementById("fileInput");
    const inputContainer = document.getElementById("image-container-Ori");

    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });


    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });


    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        handleFile(e.dataTransfer.files);
    });


    dropZone.addEventListener('click', () => {
        fileInput.click();
    });


    fileInput.addEventListener("change", function () {
        handleFile(fileInput.files);
    });


    function appendInputData(el) {
        let inputData = document.getElementById("inputDataEl");
        if (inputData) {
            inputContainer.removeChild(inputData);
        }
        inputContainer.appendChild(el);
        inputContainer.scrollIntoView(true, { behavior: 'smooth', block: "center" }); 
    }

    
    function sendDataToModel(file, type = "image") {
        let url
        let fetchObj = {
            method: "POST"
        }
        const formData = new FormData();
        formData.append('data', file); 
        if (type == "image") {
            url = "/process-image"
            fetchObj["body"] = formData
        } else {
            url = "/upload-video"
            fetchObj["body"] = formData
        }
        fetch(url, fetchObj);
    }

    function displayVideo(files) {
        const file = files[0];
        const videoURL = URL.createObjectURL(file);

        originalImageContainer.style.display = "flex";
        originalImageContainer.style.flexDirection = "column";
        originalImageContainer.style.justifyContent = "space-between";

        let videoEl = document.createElement("video");
        videoEl.src = videoURL;
        videoEl.id = "inputDataEl";
        videoEl.controls = "controls";
        appendInputData(videoEl);
    }

    function handleFile(files) {
        if (files.length > 1) {
            console.log("Accept 1 file only");
            return;
        }
        let file = files[0];
        if (file.type.startsWith('image/')) {
            displayImage(file);
            sendDataToModel(file, "image");
        } else if (file.type.startsWith('video/')) {
            displayVideo(files);
            sendDataToModel(file, "video");
        }
    }

    function displayImage(file) {
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                let imageEl = document.createElement("img");
                imageEl.src = e.target.result;
                originalImageContainer.style.display = "flex";
                originalImageContainer.style.flexDirection = "row";
                originalImageContainer.style.justifyContent = "space-between";
                imageEl.id = "inputDataEl";
                appendInputData(imageEl);
            };
            reader.readAsDataURL(file);
        }
    }
});

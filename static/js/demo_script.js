document.addEventListener("DOMContentLoaded", function () {
    const dropZone = document.getElementById("dropZone");
    const originalImage = document.getElementById("originalImage");
    const originalImageContainer = document.getElementById("originalImageContainer");
    const fileInput = document.getElementById("fileInput");
    const inputContainer = document.getElementById("image-container-Ori");
    const outputContainer = document.getElementById("image-container-Gra");
    const loader = document.getElementById("loader")

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
        let inputData1 = document.getElementById("originalImageContainer")
        if (inputData) {
            inputContainer.removeChild(inputData);
        }
        el.style.width = "300px"
        el.style.height = "200px"
        inputContainer.appendChild(el);
        inputData1.scrollIntoView({block :'center'});
    }

    function renderOutput(filename, type = "image") {
        let existEl = document.getElementById("output-el")
        if (existEl) {
            outputContainer.removeChild(existEl)
        }
        let outputEl = document.createElement(type == "image" ? "img" : "video")
        outputEl.id = "output-el"
        if (type == "video") {
            outputEl.controls = "controls";
            outputEl.type = "video/mp4"
        }
        outputEl.src = filename
        outputEl.style.width = "300px"
        outputEl.style.height = "200px"
        outputContainer.appendChild(outputEl)
    }

    function sendDataToModel(file, type = "image") {
        let existEl = document.getElementById("output-el")
        if (existEl) {
            outputContainer.removeChild(existEl)
        }
        let url
        let fetchObj = {
            method: "POST"
        }
        const formData = new FormData();

        loader.style.display = "block";

        formData.append('data', file);
        if (type == "image") {
            url = "/process-image"
            fetchObj["body"] = formData
        } else {
            type = "video"
            url = "/upload-video"
            fetchObj["body"] = formData
        }
        fetch(url, fetchObj)
            .then(res => res.json())
            .then(data => {
                let filename = data.filename
                console.log('du ma chay gium', data)
                loader.style.display = "none"
                renderOutput(filename, type)
            })
            .catch(error => {
                console.error('Error during processing', error);

                // Hide the loader in case of an error
                loader.style.display = 'none';
            });
    }


    function handleFile(files) {
        if (files.length > 1) {
            console.log("Accept 1 file only");
            return;
        }
        let file = files[0];
        let inputData = document.getElementById("inputDataEl");
        if (inputData) {
            inputContainer.removeChild(inputData);
        }
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
    function displayVideo(files) {
        const file = files[0];
        const videoURL = URL.createObjectURL(file);

        originalImageContainer.style.display = "flex";
        originalImageContainer.style.flexDirection = "row";
        originalImageContainer.style.justifyContent = "space-between";

        let videoEl = document.createElement("video");
        videoEl.src = videoURL;
        videoEl.id = "inputDataEl";
        videoEl.controls = "controls";

        appendInputData(videoEl);
    }

});

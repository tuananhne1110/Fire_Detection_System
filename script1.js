const dragImage = document.getElementById('drag-image');
const image = document.getElementById('image');

dragImage.addEventListener('dragstart', (e) => {
    e.dataTransfer.setData('text/plain', 'dragging');
});

dragImage.addEventListener('dragend', () => {
    // Handle the end of dragging here, if needed.
});

dragImage.addEventListener('dragover', (e) => {
    e.preventDefault();
});

dragImage.addEventListener('drop', () => {
    // Display the image when dropped.
    image.style.display = 'block';
});

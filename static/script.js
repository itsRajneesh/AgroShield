function previewImage(event) {
    const file = event.target.files[0];
    const previewContainer = document.getElementById("previewContainer");
    previewContainer.innerHTML = "";
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            const img = document.createElement("img");
            img.src = e.target.result;
            img.style.maxWidth = "100%";
            img.style.height = "auto";
            previewContainer.appendChild(img);
        };
        reader.readAsDataURL(file);
    }
}

function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const rightPosition = sidebar.style.right === '0px' ? '-250px' : '0px';
    sidebar.style.right = rightPosition;
}
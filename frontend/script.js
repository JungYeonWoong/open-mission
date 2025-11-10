const BASE_URL = "http://localhost:8000/api/v1";  // FastAPI 서버 주소

async function uploadImage() {
    const fileInput = document.getElementById("imageInput");
    if (fileInput.files.length === 0) {
        alert("이미지를 선택하세요!");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    const res = await fetch(`${BASE_URL}/predict/image`, {
        method: "POST",
        body: formData,
    });

    const data = await res.json();
    document.getElementById("resultBox").innerText = JSON.stringify(data, null, 2);
}


async function uploadVideo() {
    const fileInput = document.getElementById("videoInput");
    if (fileInput.files.length === 0) {
        alert("비디오 파일을 선택하세요!");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    const res = await fetch(`${BASE_URL}/predict/video`, {
        method: "POST",
        body: formData,
    });

    const data = await res.json();
    document.getElementById("resultBox").innerText = JSON.stringify(data, null, 2);
}

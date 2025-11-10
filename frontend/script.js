import { drawBoundingBoxes } from "./render.js";

const BASE_URL = "http://localhost:8000/api/v1";

async function uploadImage() {
    const fileInput = document.getElementById("imageInput");
    if (fileInput.files.length === 0) {
        alert("이미지를 선택하세요!");
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${BASE_URL}/predict/image`, {
        method: "POST",
        body: formData,
    });

    const json = await res.json();
    document.getElementById("resultBox").innerText = JSON.stringify(json, null, 2);

    // 미리보기 이미지 표시
    const preview = document.getElementById("previewImage");
    preview.src = URL.createObjectURL(file);

    // 이미지가 로드된 후 canvas 박스 렌더링
    preview.onload = () => {
        const detections = json.data?.detections || [];
        drawBoundingBoxes(preview, detections);
    };
}

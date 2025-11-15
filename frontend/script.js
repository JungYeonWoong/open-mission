// 🚨 import는 반드시 최상단!
import { drawBoundingBoxes } from "./render.js";

console.log("SCRIPT START");

const BASE_URL = "http://localhost:8000/api/v1";

window.addEventListener("DOMContentLoaded", () => {
    console.log("DOM READY!");

    /* ========================================================
     * 📸 IMAGE TAB LOGIC
     * ======================================================== */
    const imageInput = document.getElementById("imageInput");
    const imageFileName = document.getElementById("imageFileName");
    const imagePredictBtn = document.getElementById("imagePredictBtn");
    const dropArea = document.getElementById("dropArea");

    // 🔥 공용 함수: 이미지 파일을 불러왔을 때 처리하는 모든 로직
    function handleImageFile(file) {
        if (!file) return;

        // 파일명 표시
        imageFileName.textContent = file.name;

        // 이미지 미리보기
        const preview = document.getElementById("previewImage");
        preview.src = URL.createObjectURL(file);

        preview.onload = () => {
            const canvas = document.getElementById("canvas");
            canvas.width = preview.width;
            canvas.height = preview.height;

            const ctx = canvas.getContext("2d");
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            document.getElementById("resultBox").innerText =
                "이미지를 선택했습니다. 추론을 실행하세요.";
        };
    }

    // 📁 input 파일 선택 시
    imageInput?.addEventListener("change", () => {
        const file = imageInput.files[0];
        handleImageFile(file);
    });

    // 🚀 이미지 추론
    imagePredictBtn?.addEventListener("click", uploadImage);


    /* ========================================================
     * 🖱 드래그 & 드롭 업로드
     * ======================================================== */
    if (dropArea) {
        dropArea.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropArea.classList.add("dragover");
        });

        dropArea.addEventListener("dragleave", () => {
            dropArea.classList.remove("dragover");
        });

        dropArea.addEventListener("drop", (e) => {
            e.preventDefault();
            dropArea.classList.remove("dragover");

            const file = e.dataTransfer.files[0];
            if (!file) return;

            // imageInput에 강제로 파일 주입
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            imageInput.files = dataTransfer.files;

            // 이미지 업로드와 동일한 처리 실행
            handleImageFile(file);
        });
    }


    /* ========================================================
     * 🎥 VIDEO TAB LOGIC
     * ======================================================== */
    const videoInput = document.getElementById("videoInput");
    const videoFileName = document.getElementById("videoFileName");
    const videoPreview = document.getElementById("videoPreview");
    const videoPredictBtn = document.getElementById("videoPredictBtn");
    const videoLogBox = document.getElementById("videoLogBox");

    // 📁 비디오 input 선택 시
    videoInput?.addEventListener("change", () => {
        const file = videoInput.files[0];
        if (!file) return;

        videoFileName.textContent = file.name;

        videoPreview.src = URL.createObjectURL(file);
        videoPreview.load();

        videoLogBox.textContent = "비디오가 선택되었습니다. 추론을 실행하세요.";
    });

    videoPredictBtn?.addEventListener("click", uploadVideo);
});


/* ========================================================
 * 🔥 이미지 추론 API 호출
 * ======================================================== */
async function uploadImage() {
    const fileInput = document.getElementById("imageInput");

    if (fileInput.files.length === 0) {
        alert("이미지를 선택해주세요!");
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    document.getElementById("resultBox").innerText =
        "🔥 화재 감지 중입니다...";

    const res = await fetch(`${BASE_URL}/predict/image/`, {
        method: "POST",
        body: formData,
    });

    const json = await res.json();
    document.getElementById("resultBox").innerText =
        JSON.stringify(json, null, 2);

    const preview = document.getElementById("previewImage");

    preview.onload = () => {
        const detections = json.data?.detections || [];
        drawBoundingBoxes(preview, detections);
    };
}


/* ========================================================
 * 🔥 비디오 추론 API 호출 
 * ======================================================== */
async function uploadVideo() {
    const fileInput = document.getElementById("videoInput");
    const videoLogBox = document.getElementById("videoLogBox");

    if (fileInput.files.length === 0) {
        alert("비디오를 선택해주세요!");
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    videoLogBox.textContent = "🔥 비디오 화재 감지 중입니다...";

    try {
        const res = await fetch(`${BASE_URL}/predict/video`, {
            method: "POST",
            body: formData,
        });

        const json = await res.json();
        videoLogBox.textContent = JSON.stringify(json, null, 2);

    } catch (err) {
        videoLogBox.textContent = "❌ 비디오 추론 중 오류 발생: " + err;
    }
}

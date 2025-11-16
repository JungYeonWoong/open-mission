// 🚨 import는 반드시 최상단!
import { drawBoundingBoxes } from "./render.js";
import { renderFrequencyChart } from "./chart.js";

console.log("SCRIPT START");

const BASE_URL = "http://localhost:8000/api/v1";

// 마지막 비디오 추론 결과의 detection frequency 저장
let lastDetectionFrequency = null;

window.addEventListener("DOMContentLoaded", () => {
    console.log("DOM READY!");

    /* ========================================================
     * 📸 IMAGE TAB LOGIC
     * ======================================================== */
    const imageInput = document.getElementById("imageInput");
    const imageFileName = document.getElementById("imageFileName");
    const imagePredictBtn = document.getElementById("imagePredictBtn");
    const dropArea = document.getElementById("dropArea");

    // 공용 함수: 이미지 파일을 불러왔을 때 처리하는 모든 로직
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
            canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);

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
    const showFreqChartBtn = document.getElementById("showFreqChartBtn");

    // 📁 비디오 input 선택 시
    videoInput?.addEventListener("change", () => {
        const file = videoInput.files[0];
        if (!file) return;

        videoFileName.textContent = file.name;

        videoPreview.src = URL.createObjectURL(file);
        // videoPreview.load();
        videoPreview.onloadedmetadata = () => {
            console.log("Video metadata loaded");
            videoPreview.play();
        };

        videoLogBox.textContent = "비디오가 선택되었습니다. 추론을 실행하세요.";
    });

    videoPredictBtn?.addEventListener("click", uploadVideo);

    showFreqChartBtn?.addEventListener("click", () => {
        if (!lastDetectionFrequency) {
            alert("먼저 비디오 추론을 실행해주세요.");
            return;
        }

        renderFrequencyChart(lastDetectionFrequency);
    });
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

    // ================================
    // 📌 원본 이미지 미리보기 + canvas bbox 그리기
    // ================================
    const preview = document.getElementById("previewImage");

    preview.onload = () => {
        const detections = json.data?.detections || [];
        drawBoundingBoxes(preview, detections);
    };

    // previewImage에 원본 이미지 표시
    preview.src = URL.createObjectURL(file);

    // ================================
    // 🔥 추론 결과 이미지(resultImage) 표시
    // ================================
    const resultImg = document.getElementById("resultImage");
    const savedPath = json.data?.saved_result_path;

    if (savedPath) {
        resultImg.src = savedPath;   // 서버에 저장된 추론 결과 이미지 로드
    } else {
        resultImg.src = "";          // 결과 없음 → 이미지 제거
    }

}


/* ========================================================
 * 🔥 비디오 추론 API 호출 
 * ======================================================== */
async function uploadVideo() {
    const fileInput = document.getElementById("videoInput");
    const videoLogBox = document.getElementById("videoLogBox");
    const videoPreview = document.getElementById("videoPreview");
    const videoResult = document.getElementById("videoResult");

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

        // ==============================
        // 📊 Detection Frequency 저장 + 버튼 활성화
        // ==============================
        const freqData = json.data?.detection_frequency;
        console.log("freqData =", freqData);

        if (freqData) {
            lastDetectionFrequency = freqData;
            showFreqChartBtn.disabled = false; // 버튼 활성화
        }

        // ==============================
        // ⭐ 추론 결과 비디오 표시
        // ==============================
        const resultVideoPath = json.data?.output_video;
        console.log("output_video =", resultVideoPath);


        if (resultVideoPath) {
            // 서버가 “/static/...mp4” 형태 반환하는 경우 그대로 넣으면 됨
            videoResult.src = resultVideoPath;
            videoResult.load();
            videoResult.play();
        } else {
            videoResult.src = "";
        }

    } catch (err) {
        videoLogBox.textContent = "❌ 비디오 추론 중 오류 발생: " + err;
        videoResult.src = "";
    }
}
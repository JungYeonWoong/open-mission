// 🚨 import는 반드시 최상단!
import { drawBoundingBoxes } from "./render.js";

console.log("SCRIPT START");

// DOMContentLoaded 디버그 로그
window.addEventListener("DOMContentLoaded", () => {
    console.log("DOM READY!", document.getElementById("imageInput"));
});

const BASE_URL = "http://localhost:8000/api/v1";

window.addEventListener("DOMContentLoaded", () => {
    const imageInput = document.getElementById("imageInput");
    const imagePredictBtn = document.getElementById("imagePredictBtn");

    /* -----------------------------------
     * 🔍 파일 선택 시 자동 미리보기 표시
     * ----------------------------------- */
    imageInput.addEventListener("change", () => {
        const file = imageInput.files[0];
        if (!file) return;

        const preview = document.getElementById("previewImage");
        preview.src = URL.createObjectURL(file);

        // 이미지 로드 후 캔버스 초기화
        preview.onload = () => {
            const canvas = document.getElementById("canvas");
            canvas.width = preview.width;
            canvas.height = preview.height;

            const ctx = canvas.getContext("2d");
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            document.getElementById("resultBox").innerText =
                "이미지를 선택했습니다. 추론을 실행하세요.";
        };
    });

    /* -----------------------------------
     * 🚀 이미지 추론 버튼 클릭 이벤트
     * ----------------------------------- */
    imagePredictBtn.addEventListener("click", uploadImage);
});

/* -----------------------------------
 * 🚀 이미지 추론하기 (서버 호출)
 * ----------------------------------- */
async function uploadImage() {
    const fileInput = document.getElementById("imageInput");

    if (fileInput.files.length === 0) {
        alert("이미지를 선택해주세요!");
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    // 서버 요청
    const res = await fetch(`${BASE_URL}/predict/image`, {
        method: "POST",
        body: formData,
    });

    const json = await res.json();

    // JSON 출력
    document.getElementById("resultBox").innerText =
        JSON.stringify(json, null, 2);

    // 박스 렌더링
    const preview = document.getElementById("previewImage");

    preview.onload = () => {
        const detections = json.data?.detections || [];
        drawBoundingBoxes(preview, detections);
    };
}

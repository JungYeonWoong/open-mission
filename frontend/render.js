export function drawBoundingBoxes(imageElement, detections) {
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    canvas.width = imageElement.width;
    canvas.height = imageElement.height;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    detections.forEach(det => {
        const { x1, y1, x2, y2 } = det.bbox;

        // 색상 (라벨별 색상 고정 가능 — 지금은 초록)
        const color = "#00ff00";

        // 1) Bounding box 그리기
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // 2) Label + Confidence 텍스트 생성
        const label = det.label;
        const conf = (det.confidence * 100).toFixed(1);  // 0.87 → 87.0%

        const text = `${label} ${conf}%`;

        ctx.font = "16px Arial";
        ctx.textBaseline = "top";

        // 3) 텍스트 배경 박스
        const textWidth = ctx.measureText(text).width;
        const textHeight = 16;

        ctx.fillStyle = color;                 // 배경색
        ctx.fillRect(x1, y1 - textHeight, textWidth + 6, textHeight + 4);

        // 4) 텍스트 그리기
        ctx.fillStyle = "#000000";             // 글자색: 검정
        ctx.fillText(text, x1 + 3, y1 - textHeight + 2);
    });
}

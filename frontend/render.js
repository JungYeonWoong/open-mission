export function drawBoundingBoxes(imageElement, detections) {
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    // canvas 크기를 이미지와 동일하게 설정
    canvas.width = imageElement.width;
    canvas.height = imageElement.height;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    detections.forEach(det => {
        const { x1, y1, x2, y2 } = det.bbox;

        ctx.strokeStyle = "#00ff00";  // 초록색 박스
        ctx.lineWidth = 2;

        ctx.strokeRect(
            x1, 
            y1, 
            x2 - x1, 
            y2 - y1
        );
    });
}

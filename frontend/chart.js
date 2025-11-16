let freqChartInstance = null;

export function renderFrequencyChart(freq) {
    const canvas = document.getElementById("freqChart");

    if (!canvas) {
        console.warn("freqChart canvas not found.");
        return;
    }

    const ctx = canvas.getContext("2d");

    // 기존 차트 제거 (중복 생성 방지)
    if (freqChartInstance !== null) {
        freqChartInstance.destroy();
    }

    // Chart.js 그래프 생성
    freqChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["Smoke_White", "Smoke_Grey", "Smoke_Black", "Fire"],
            datasets: [
                {
                    label: "Detection Frequency",
                    data: [
                        freq[0] || 0,
                        freq[1] || 0,
                        freq[2] || 0,
                        freq[3] || 0
                    ],
                    backgroundColor: ["#d9d9d9", "#808080", "#333333", "#ff0000"]
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: { display: true, text: "Frame Count" }
                }
            }
        }
    });
}

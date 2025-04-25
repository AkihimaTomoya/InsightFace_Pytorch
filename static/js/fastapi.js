
const video = document.getElementById('videoStream');
const canvasOverlay = document.getElementById('canvasOverlay');
const ctx = canvasOverlay.getContext('2d');
const connectionStatusEl = document.getElementById('connectionStatus');

let ws = null;
let realtimeStream = null;
let processingFrame = false;
let faceLocations = [];  // Mảng vị trí khuôn mặt - cập nhật liên tục
let recognitionData = {}; // Dữ liệu nhận dạng khuôn mặt - cập nhật mỗi 2 giây
let config = { show_bbox: true, show_label: true }; // Default config

// Kết nối WebSocket
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('WebSocket connection established');
        connectionStatusEl.textContent = 'Connected';
        connectionStatusEl.style.color = '#4caf50';
    };

    ws.onclose = () => {
        console.log('WebSocket connection closed');
        connectionStatusEl.textContent = 'Disconnected - Retrying...';
        connectionStatusEl.style.color = '#f44336';
        // Thử kết nối lại sau 3 giây
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        connectionStatusEl.textContent = 'Connection Error';
        connectionStatusEl.style.color = '#f44336';
    };

    ws.onmessage = (event) => {
        try {
            const message = JSON.parse(event.data);
            handleWebSocketMessage(message);
        } catch (e) {
            console.error('Error parsing message:', e);
        }
    };
}

// Xử lý từ server
function handleWebSocketMessage(message) {
    switch (message.type) {
        case 'face_locations':
            console.log('Received face locations:', message.data);
            faceLocations = message.data;
            break;

        case 'recognition_data':
            console.log('Received recognition data:', message.data);
            recognitionData = message.data;
            document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
            break;

        case 'config':
            console.log('Received config:', message.data);
            config = message.data;
            break;

        default:
            console.log('Unknown message type:', message.type);
    }
}

// Cấu hình video stream cho realtime
function setupRealtimeVideo() {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            realtimeStream = stream;
            video.srcObject = stream;
            video.play();

            // Đảm bảo canvas có kích thước phù hợp với video
            video.addEventListener('loadedmetadata', () => {
                canvasOverlay.width = video.videoWidth;
                canvasOverlay.height = video.videoHeight;
                startAnimation();
                processVideoFrames(); // Bắt đầu xử lý frame
            });
        })
        .catch(err => {
            console.error("Error accessing camera in realtime mode: " + err);
            document.getElementById('status').textContent = "Camera error: " + err.message;
        });
}

// Hàm vẽ bounding boxes lên canvas
function drawFaceBoxes() {
    // Xóa canvas
    ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);

    // Hiển thị status
    const statusEl = document.getElementById('status');
    if (faceLocations.length === 0) {
        statusEl.textContent = "No faces detected";
    } else {
        statusEl.textContent = `Detected ${faceLocations.length} face(s)`;
    }

    // Vẽ các khuôn mặt
    faceLocations.forEach(face => {
        const [x1, y1, x2, y2] = face.bbox;
        const width = x2 - x1;
        const height = y2 - y1;
        const faceId = face.id;

        // Lấy thông tin nhận dạng nếu có
        const recognition = recognitionData[faceId] || { name: "Processing...", confidence: 0 };
        const name = recognition.name;
        const confidence = recognition.confidence;

        // Vẽ bbox nếu được bật
        if (config.show_bbox) {
            ctx.strokeStyle = confidence >= 0.8 ? 'lime' : 'red';
            ctx.lineWidth = 2;
            ctx.strokeRect(x1, y1, width, height);
        }

        // Vẽ label nếu được bật
        if (config.show_label) {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
            ctx.fillRect(x1, y1 - 20, width, 20);

            ctx.fillStyle = 'white';
            ctx.font = '14px Arial';
            const displayText = confidence >= 0.8 ?
                `${name} (${(confidence * 100).toFixed(0)}%)` :
                (name === "Processing..." ? name : 'Unknown');
            ctx.fillText(displayText, x1 + 5, y1 - 5);
        }
    });
}

// Animation loop
function startAnimation() {
    function animate() {
        drawFaceBoxes();
        requestAnimationFrame(animate);
    }
    animate();
}

// Hàm xử lý khung hình video và gửi lên server
function processVideoFrames() {
    if (!ws || ws.readyState !== WebSocket.OPEN || processingFrame) {
        // Thử lại sau 300ms nếu không sẵn sàng
        setTimeout(processVideoFrames, 300);
        return;
    }

    processingFrame = true;

    // Tạo canvas để capture frame hiện tại
    const captureCanvas = document.createElement('canvas');
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    const captureCtx = captureCanvas.getContext('2d');
    captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

    // Chuyển đổi thành dạng base64 và gửi lên server
    const dataURL = captureCanvas.toDataURL('image/jpeg', 0.85);
    ws.send(dataURL);

    // Lên lịch xử lý frame tiếp theo sau 200ms
    setTimeout(() => {
        processingFrame = false;
        processVideoFrames();
    }, 200);
}

// Thiết lập và khởi động khi trang load
document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
    setupRealtimeVideo();
});

import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";
const demosSection = document.getElementById("demos");
let poseLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;
const videoHeight = "360px";
const videoWidth = "480px";

import kNear from "./knear.js"

const k = 5
const machine = new kNear(k);

const createPoseLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
            delegate: "GPU"
        },
        runningMode: runningMode,
        numPoses: 1,
        min_pose_detection_confidence: 0.8
    });
    demosSection.classList.remove("invisible");
};
createPoseLandmarker();

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);

const hasGetUserMedia = () => { var _a; return !!((_a = navigator.mediaDevices) === null || _a === void 0 ? void 0 : _a.getUserMedia); };

if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);
}
else {
    console.warn("getUserMedia() is not supported by your browser");
}

function enableCam(event) {
    if (!poseLandmarker) {
        console.log("Wait! poseLandmaker not loaded yet.");
        return;
    }
    if (webcamRunning === true) {
        webcamRunning = false;
        enableWebcamButton.innerText = "ENABLE PREDICTIONS";
    }
    else {
        webcamRunning = true;
        enableWebcamButton.innerText = "DISABLE PREDICTIONS";
    }

    const constraints = {
        video: true
    };

    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    });
}

let poseArray = [];
fetch('data.json')
    .then(response => response.json())
    .then(data => {
        for (let i = 0; i < data.length; i++) {
            const item = data[i];
            machine.learn(item.pose, item.tag)
            poseArray.push([item.pose, item.tag])
        }
    })
    .catch(error => console.error('Error reading data:', error));

//zet variabelen
let lastVideoTime = -1;
const scoreText = document.getElementById("scoreText");
const positionText = document.getElementById("pos");

let score = 0;
let getRep = false;
let repTimer = 0;

let yourScore = 0;
let highScore = 0;

async function predictWebcam() {
    canvasElement.style.height = videoHeight;
    video.style.height = videoHeight;
    canvasElement.style.width = videoWidth;
    video.style.width = videoWidth;
    // Now let's start detecting the stream.
    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await poseLandmarker.setOptions({ runningMode: "VIDEO" });
    }
    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        poseLandmarker.detectForVideo(video, startTimeMs, (result) => {
            canvasCtx.save();
            canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
            for (const landmark of result.landmarks) {
                drawingUtils.drawLandmarks(landmark, {
                    radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1)
                });
                drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);

                //de pose die je momenteel aanhoud
                const flattenedArray = landmark.flatMap(obj => [obj.x, obj.y, obj.z]);

                //de prediction van de ai
                let prediction = machine.classify(flattenedArray)

                //zet de text en bereken de score.
                if (prediction === "Squat") {
                    getRep = true;
                    positionText.innerHTML = "Current position: Squat";
                }else if (prediction === "Standing") {
                    if (getRep) {
                        score++;
                        repTimer = 150;
                        scoreText.innerHTML = "Score: "+score+ " -- Highscore: "+highScore;
                        getRep = false;
                    }
                    positionText.innerHTML = "Current Position: Standing";
                }else{
                    positionText.innerHTML = "Current Position: Middle";
                }

                //zet de timer en bereken of je een highscore heb berekent.
                if (repTimer > 0) {
                    repTimer--;
                }
                if (repTimer <= 0) {
                    yourScore = score;
                    if (yourScore > highScore) {
                        highScore = yourScore;
                    }
                    score = 0;
                    scoreText.innerHTML = "Score: "+score+ " -- Highscore: "+highScore;
                }
            }
            canvasCtx.restore();
        });
    }
    if (webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
    }

}
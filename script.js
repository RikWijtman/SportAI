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
        numPoses: 2
    });
    demosSection.classList.remove("invisible");
};
createPoseLandmarker();

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);

const testButtonSD = document.getElementById("buttonSquatDown");
const testButtonSU = document.getElementById("buttonSquatUp");
const testButtonSM = document.getElementById("buttonSquatMiddle");
const testAcc = document.getElementById("buttonAcc");

const hasGetUserMedia = () => { var _a; return !!((_a = navigator.mediaDevices) === null || _a === void 0 ? void 0 : _a.getUserMedia); };

if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);

    if (testButtonSU != null) {
        testButtonSD.addEventListener("click", enableTestingSD)
        testButtonSU.addEventListener("click", enableTestingSU)
        testButtonSM.addEventListener("click", enableTestingSM)
        testAcc.addEventListener("click", testAccuracy)
    }
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

let lastVideoTime = -1;
let testing = 0;
let testSort = "";
const scoreText = document.getElementById("scoreText");
const positionText = document.getElementById("pos");

let score = 0;
let getRep = false;
let repTimer = 0;

let yourScore = 0;
let highScore = 0;

function enableTestingSD() {
    if (testing <= 0) {
        testing = 150;
        testSort = "SD";
    }
}

function enableTestingSU() {
    if (testing <= 0) {
        testing = 150;
        testSort = "SU";
    }
}

function enableTestingSM() {
    if (testing <= 0) {
        testing = 150;
        testSort = "SM";
    }
}

let train;
let test;
let testingAcc;
let testTimer;
function testAccuracy() {
    testTimer = 50;
    testingAcc = true;
}

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

                const flattenedArray = landmark.flatMap(obj => [obj.x, obj.y, obj.z]);
                if (testing > 0) {
                    testing--;


                    if (testSort === "SD") {
                        if (testing < 100) {
                            poseArray.push({pose: flattenedArray, label:"Down"});
                            console.log(poseArray);
                            console.log(flattenedArray);
                            machine.learn(flattenedArray, 'Down')

                            testButtonSD.innerHTML = testing;
                        }else{
                            testButtonSD.innerHTML = "Get ready";
                        }
                    }

                    if (testSort === "SM") {
                        if (testing < 100) {
                            poseArray.push({pose: flattenedArray, label:"Middle"});
                            console.log(poseArray);
                            console.log(poseArray);
                            machine.learn(flattenedArray, 'Middle')

                            testButtonSM.innerHTML = testing;
                        }else{
                            testButtonSM.innerHTML = "Get ready";
                        }
                    }

                    if (testSort === "SU") {
                        if (testing < 100) {
                            poseArray.push({pose: flattenedArray, label:"Up"});
                            console.log(poseArray);
                            console.log(poseArray);
                            machine.learn(flattenedArray, 'Up')

                            testButtonSU.innerHTML = testing;
                        }else{
                            testButtonSU.innerHTML = "Get ready";
                        }
                    }

                }else{
                    if (testButtonSU != null) {
                        testButtonSD.innerHTML = "Test squat down";
                        testButtonSM.innerHTML = "Test squat middle";
                        testButtonSU.innerHTML = "Test squat up";
                    }
                }



                let prediction = machine.classify(flattenedArray)

                if (prediction === "Down") {
                    getRep = true;
                    positionText.innerHTML = "Current position: Down";
                }else if (prediction === "Up") {
                    if (getRep) {
                        score++;
                        repTimer = 150;
                        scoreText.innerHTML = "Score: "+score+ " -- Highscore: "+highScore;
                        getRep = false;
                    }
                    positionText.innerHTML = "Current Position: Up";
                }else{
                    positionText.innerHTML = "Current Position: Middle";
                }

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



                //Test accuracy gedeelte:
                if (testTimer > 0) {
                    testTimer--
                    testAcc.innerHTML = testTimer;
                }

                if (testingAcc && testTimer <= 0) {
                    poseArray.sort(() => (Math.random() - 0.5))

                    train = poseArray.slice(0, Math.floor(poseArray.length * 0.8))
                    test = poseArray.slice(Math.floor(poseArray.length * 0.8) + 1)

                    const testpose = {pose:[0.5078023672103882, 0.2812376022338867, -1.0536397695541382, 0.522529661655426, 0.24578744173049927, -1.0115286111831665, 0.5334699153900146, 0.24574893712997437, -1.0115286111831665, 0.5437833666801453, 0.24619174003601074, -1.0115286111831665, 0.48857754468917847, 0.2474738359451294, -1.006372094154358, 0.4773235023021698, 0.2486574649810791, -1.006372094154358, 0.46681660413742065, 0.2505979537963867, -1.006372094154358, 0.5591275691986084, 0.2631303071975708, -0.6999915242195129, 0.4533641040325165, 0.26730239391326904, -0.6759279370307922, 0.5305887460708618, 0.32228487730026245, -0.9324626326560974, 0.4903390407562256, 0.32459795475006104, -0.9264467358589172, 0.6690473556518555, 0.5043743252754211, -0.4533401131629944, 0.35920727252960205, 0.5117442607879639, -0.4425974488258362, 0.7377007603645325, 0.8102378845214844, -0.38286828994750977, 0.2813761234283447, 0.7977057695388794, -0.35042545199394226, 0.7799532413482666, 1.0592799186706543, -0.702999472618103, 0.1997189223766327, 1.0689644813537598, -0.6389732360839844, 0.8185980916023254, 1.1550288200378418, -0.7889407277107239, 0.15977761149406433, 1.1529178619384766, -0.7154609560966492, 0.7944815158843994, 1.1541860103607178, -0.8937890529632568, 0.1870327889919281, 1.164359450340271, -0.828473687171936, 0.776187002658844, 1.1220225095748901, -0.7584315538406372, 0.2075275480747223, 1.13444983959198, -0.6952647566795349, 0.5834251642227173, 1.0504928827285767, 0.00010658730025170371, 0.40705716609954834, 1.0367867946624756, 0.00544518418610096, 0.5807042717933655, 1.497401237487793, -0.21904277801513672, 0.3992069363594055, 1.4853825569152832, -0.19411981105804443, 0.576133131980896, 1.8452324867248535, 0.294348806142807, 0.4057071805000305, 1.8807235956192017, 0.17005625367164612, 0.5795939564704895, 1.896641492843628, 0.3418313264846802, 0.40694114565849304, 1.9425899982452393, 0.1949792206287384, 0.5483196973800659, 1.9639053344726562, 0.08153676241636276, 0.4128583073616028, 2.0030276775360107, -0.12225143611431122]
                        , label:"Up"}
                    const pred = machine.classify(testpose.pose)
                    console.log(`Ik voorspelde: ${pred}. Het correcte antwoord is: ${testpose.label}`)

                    testingAcc = false;

                    testAcc.innerHTML = "test acc";
                }
            }
            canvasCtx.restore();
        });
    }
    if (webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
    }

}
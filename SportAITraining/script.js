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
        numPoses: 1
    });
    demosSection.classList.remove("invisible");
};
createPoseLandmarker();

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);

//zet buttons van het html document
const testButtonSD = document.getElementById("buttonSquatDown");
const testButtonSU = document.getElementById("buttonSquatUp");
const testButtonSM = document.getElementById("buttonSquatMiddle");
const testAcc = document.getElementById("buttonAcc");

const hasGetUserMedia = () => { var _a; return !!((_a = navigator.mediaDevices) === null || _a === void 0 ? void 0 : _a.getUserMedia); };

if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);

    //voeg eventlisteners toe aan de knoppen
    if (testButtonSU != null) {
        testButtonSU.addEventListener("click", () => {
            enableTesting("SU");
        });
    }else{
        console.log("TestButtonUp can't be found");
    }
    if (testAcc != null) {
        testAcc.addEventListener("click", testAccuracy)
    }else{
        console.log("TestButtonAccuracy can't be found");
    }
    if (testButtonSM != null) {
        testButtonSM.addEventListener("click", () => {
            enableTesting("SM");
        });
    }else{
        console.log("TestButtonMiddle can't be found");
    }
    if (testButtonSD != null) {
        testButtonSD.addEventListener("click", () => {
            enableTesting("SD");
        });
    }else{
        console.log("TestButtonDown can't be found");
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
            poseArray.push([item.pose, item.tag])
        }
        //splitten van train en test data
        poseArray.sort(() => (Math.random() - 0.5))

        train = poseArray.slice(0, Math.floor(poseArray.length * 0.8))
        test = poseArray.slice(Math.floor(poseArray.length * 0.8) + 1)

        //alles uit train wordt geleerd aan de ML
        //alles van test word later gebruikt om accuracy te testen
        for (let i = 0; i < train.length; i++) {
            const item = train[i];
            machine.learn(item[0], item[1])
        }
    })
    .catch(error => console.error('Error reading data:', error));

//zetten variabelen en elementen uit het html document
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

//functies die afgaan wanneer je een bepaalde pose test
function enableTesting(sort) {
    if (testing <= 0) {
        testing = 150;
        testSort = sort;
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

                //de pose die je momenteel aanhoud
                const flattenedArray = landmark.flatMap(obj => [obj.x, obj.y, obj.z]);

                //Check of we aan het testen zijn
                if (testing > 0) {
                    testing--;

                    //Welke positie testen we?
                    if (testSort === "SD") {
                        if (testing < 100) {
                            poseArray.push({pose: flattenedArray, label:"Squat"});
                            console.log(poseArray);
                            console.log(flattenedArray);
                            machine.learn(flattenedArray, 'Squat')

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
                            poseArray.push({pose: flattenedArray, label:"Standing"});
                            console.log(poseArray);
                            console.log(poseArray);
                            machine.learn(flattenedArray, 'Standing')

                            testButtonSU.innerHTML = testing;
                        }else{
                            testButtonSU.innerHTML = "Get ready";
                        }
                    }

                }else{
                    if (testButtonSU != null) {
                        testButtonSD.innerHTML = "Test Squat";
                        testButtonSM.innerHTML = "Test Middle";
                        testButtonSU.innerHTML = "Test Standing";
                    }
                }


                //Wat is de prediction?
                let prediction = machine.classify(flattenedArray)

                //Zetten van tekst en score bijwerken
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

                //Checken of score word gereset
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

                    let totalTestPoses = 0;
                    let correctPredictions = 0;

                    for (let i = 0; i < 10; i++) {
                        let randomPose = Math.round(Math.random() * (test.length - 1));
                        console.log(test[randomPose]);

                        const testpose = {"pose": test[randomPose][0], "tag": test[randomPose][1]};
                        const pred = machine.classify(testpose.pose);
                        console.log(`Ik voorspelde: ${pred}. Het correcte antwoord is: ${testpose.tag}`);

                        totalTestPoses++;
                        if (pred === testpose.tag) {
                            correctPredictions++;
                        }
                    }

                    //berekenen accuracy, word gezet in de tekst
                    let accuracy = correctPredictions / totalTestPoses;
                    testAcc.innerHTML = Math.round(accuracy * 100) + "%";

                    testingAcc = false;

                }
            }
            canvasCtx.restore();
        });
    }
    if (webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
    }

}
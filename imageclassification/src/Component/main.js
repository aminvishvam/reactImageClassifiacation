import React, { useState } from 'react';
import * as tf from "@tensorflow/tfjs";
// import Webcam from "react-webcam";
// import handData from "./dataCollection";

// var yesSemple = 0, noSamples = 0, callMeSamples = 0, peaceSamples = 0;

class handData {
    constructor() {
        this.labels = []
    }

    addExample(example, label) {
        if (this.xs == null) {
            this.xs = tf.keep(example);
            this.labels.push(label);
        } else {
            const oldX = this.xs;
            this.xs = tf.keep(oldX.concat(example, 0));
            this.labels.push(label);
            oldX.dispose();
        }
    }

    encodeLabels(numClasses) {
        for (var i = 0; i < this.labels.length; i++) {
            if (this.ys == null) {
                this.ys = tf.keep(tf.tidy(
                    () => {
                        return tf.oneHot(
                            tf.tensor1d([this.labels[i]]).toInt(), numClasses)
                    }));
            } else {
                const y = tf.tidy(
                    () => {
                        return tf.oneHot(
                            tf.tensor1d([this.labels[i]]).toInt(), numClasses)
                    });
                const oldY = this.ys;
                this.ys = tf.keep(oldY.concat(y, 0));
                oldY.dispose();
                y.dispose();
            }
        }
    }

}
class Webcam {
    /**
     * @param {HTMLVideoElement} webcamElement A HTMLVideoElement representing the
     *     webcam feed.
     */
    constructor(webcamElement) {
        this.webcamElement = webcamElement;
    }

    /**
     * Captures a frame from the webcam and normalizes it between -1 and 1.
     * Returns a batched image (1-element batch) of shape [1, w, h, c].
     */


    capture() {
        return tf.tidy(() => {
            // Reads the image as a Tensor from the webcam <video> element.
            const webcamImage = tf.browser.fromPixels(this.webcamElement);

            const reversedImage = webcamImage.reverse(1);

            // Crop the image so we're using the center square of the rectangular
            // webcam.
            const croppedImage = this.cropImage(reversedImage);

            // Expand the outer most dimension so we have a batch size of 1.
            const batchedImage = croppedImage.expandDims(0);

            // Normalize the image between -1 and 1. The image comes in between 0-255,
            // so we divide by 127 and subtract 1.
            return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
        });
    }

    /**
     * Crops an image tensor so we get a square image with no white space.
     * @param {Tensor4D} img An input image Tensor to crop.
     */


    cropImage(img) {
        const size = Math.min(img.shape[0], img.shape[1]);
        const centerHeight = img.shape[0] / 2;
        const beginHeight = centerHeight - (size / 2);
        const centerWidth = img.shape[1] / 2;
        const beginWidth = centerWidth - (size / 2);
        return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
    }

    /**
     * Adjusts the video size so we can make a centered square crop without
     * including whitespace.
     * @param {number} width The real width of the video element.
     * @param {number} height The real height of the video element.
     */


    adjustVideoSize(width, height) {
        const aspectRatio = width / height;
        if (width >= height) {
            this.webcamElement.width = aspectRatio * this.webcamElement.height;
        } else if (width < height) {
            this.webcamElement.height = this.webcamElement.width / aspectRatio;
        }
    }

    async setup() {
        return new Promise((resolve, reject) => {
            navigator.getUserMedia = navigator.getUserMedia ||
                navigator.webkitGetUserMedia || navigator.mozGetUserMedia ||
                navigator.msGetUserMedia;
            if (navigator.getUserMedia) {
                navigator.getUserMedia(
                    { video: { width: 224, height: 224 } },
                    stream => {
                        this.webcamElement.srcObject = stream;
                        this.webcamElement.addEventListener('loadeddata', async () => {
                            this.adjustVideoSize(
                                this.webcamElement.videoWidth,
                                this.webcamElement.videoHeight);
                            resolve();
                        }, false);
                    },
                    error => {
                        reject(error);
                    });
            } else {
                reject();
            }
        });
    }
}

function Main() {
    let mobilenet;
    const [yesSemple, setYes] = useState(0);
    const [noSamples, setNo] = useState(0);
    const [callMeSamples, setCall] = useState(0);
    const [peaceSamples, setPeace] = useState(0);
    // const webcam = new Webcam();

    const webcamRef = React.useRef(null);
    async function loadMobilenet() {
        mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
        const layer = mobilenet.getLayer('conv_pw_13_relu');
        return tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
    }

    const capture = (id) => (onClick) => {

        // Do something
        switch (id) {
            case "1":
                setYes(yesSemple + 1)
                break;
            case "2":
                setNo(noSamples + 1)
                break;
            case "3":
                setCall(callMeSamples + 1)
                break;
            case "4":
                setPeace(peaceSamples + 1)
                break;
        }
        const label = parseInt(id);
        const image = tf.browser.fromPixels(webcamRef.current.getScreenshot())
        console.log(image);
        // handData.addExample(mobilenet.predict(image), label);
    }

    // async function init() {
    //     const image = webcamRef.current.getScreenshot();
    //     mobilenet = await loadMobilenet();
    //     tf.tidy(() => mobilenet.predict(image));
    // }
    // init();

    const videoConstraints = {
        width: 400,
        height: 400,
        facingMode: "user"
    }
    return (
        <div>
            <header>
                <Webcam
                    audio={false}
                    height={400}
                    ref={webcamRef}
                    screenshotFormat="image/jpeg"
                    width={400}
                    videoConstraints={videoConstraints}
                />
				{/* <video autoplay playsinline muted id="wc" width="300" height="300"></video> */}
                <br />
                <button
                    onClick={capture("1")}>
                    Yes
                </button>
                <button
                    onClick={capture("2")}>
                    No
                </button>
                <button
                    onClick={capture("3")}>
                    Callme
                </button>
                <button
                    onClick={capture("4")}>
                    Peace
                </button>

                <div id="yes">{yesSemple}</div>
                <div id="no">{noSamples}</div>
                <div id="callme">{callMeSamples}</div>
                <div id="peace">{peaceSamples}</div>

            </header>
        </div>
    )
};

export default Main
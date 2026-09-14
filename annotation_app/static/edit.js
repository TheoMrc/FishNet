// static/js/edit_frame.js

// Access the variables passed from the template
// imageUrl, annotations, saveUrl are available from the template

let selectedFish = null;
let selectedPointIndex = null;
let isDragging = false;
let hoverFishIndex = null;
let hoverPointIndex = null;

let mode = 'head'; // Modes: 'head' or 'midline'
let currentFishIndex = 0; // Used in midline mode for sequential zooming

let undoStack = [];
let redoStack = [];

// New variables for head movement in midline mode
let selectedHead = false;
let hoverOverHead = false;

// Function to reorder annotations based on midline presence
function reorderAnnotations() {
    annotations.sort(function (a, b) {
        const aHasMidline = a.midline_points && a.midline_points.length > 1;
        const bHasMidline = b.midline_points && b.midline_points.length > 1;

        if (aHasMidline === bHasMidline) {
            return 0; // Keep original order if both have or don't have midlines
        } else if (!aHasMidline && bHasMidline) {
            return -1; // a comes before b
        } else {
            return 1; // b comes before a
        }
    });
}

// Call the function to reorder annotations upon loading
reorderAnnotations();

// Initialize currentFishIndex and selectedFish
currentFishIndex = 0;
selectedFish = null; // In midline mode, selectedFish is null

// Get canvas and context
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Variables for scaling
let imgWidth, imgHeight;
let scale = 1;
let offsetX = 0;
let offsetY = 0;

// Load the image
const img = new Image();
img.src = imageUrl;
img.onload = function () {
    imgWidth = img.width;
    imgHeight = img.height;

    // Set canvas size to image size
    canvas.width = imgWidth;
    canvas.height = imgHeight;

    // Fit the image to the canvas while maintaining aspect ratio
    fitImageToCanvas();

    drawAnnotations();
};

function swapPoints(points, index1, index2) {
    const temp = points[index1];
    points[index1] = points[index2];
    points[index2] = temp;
}

function fitImageToCanvas() {
    const container = document.getElementById('canvas-container');
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;

    // Calculate the scale to fit the image into the container
    const scaleX = containerWidth / imgWidth;
    const scaleY = containerHeight / imgHeight;
    scale = Math.min(scaleX, scaleY);

    // Calculate offsets to center the image
    offsetX = (containerWidth - imgWidth * scale) / 2;
    offsetY = (containerHeight - imgHeight * scale) / 2;

    // Set canvas size to match the container size
    canvas.width = containerWidth;
    canvas.height = containerHeight;
}

// Helper functions to convert between data coordinates and canvas coordinates
function dataToCanvasX(dataX) {
    return dataX * scale + offsetX;
}

function dataToCanvasY(dataY) {
    return dataY * scale + offsetY;
}

function canvasToDataX(canvasX) {
    return (canvasX - offsetX) / scale;
}

function canvasToDataY(canvasY) {
    return (canvasY - offsetY) / scale;
}

function drawPlusSign(x, y, size, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x - size, y);
    ctx.lineTo(x + size, y);
    ctx.moveTo(x, y - size);
    ctx.lineTo(x, y + size);
    ctx.stroke();
}

// Load the skeleton image
const skelImg = new Image();
if (skelImageUrl) {
    skelImg.src = skelImageUrl;
}

let isSkelBtnPressed = false;

// Function to redraw the canvas
function drawAnnotations() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = false;

    // Draw the image
    ctx.drawImage(img, offsetX, offsetY, imgWidth * scale, imgHeight * scale);

    if (isSkelBtnPressed) {
        // Set globalCompositeOperation to 'lighter' to ignore black pixels
        ctx.globalCompositeOperation = 'lighter';
        ctx.drawImage(skelImg, offsetX, offsetY, imgWidth * scale, imgHeight * scale);
        // Reset globalCompositeOperation to default
        ctx.globalCompositeOperation = 'source-over';
    }

    // Draw annotations based on the current mode
    if (mode === 'head') {
        drawHeads();
    } else if (mode === 'midline') {
        drawMidlines();
    }

    // Draw hover feedback lines
    if (hoverFishIndex !== null || hoverPointIndex !== null || hoverOverHead) {
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.lineWidth = 1;
        ctx.beginPath();

        let hoverX, hoverY;
        if (mode === 'head') {
            if (hoverFishIndex !== null) {
                const fish = annotations[hoverFishIndex];
                hoverX = dataToCanvasX(fish.midline_points[0].y);
                hoverY = dataToCanvasY(fish.midline_points[0].x);
            } else {
                hoverX = lastMouseX;
                hoverY = lastMouseY;
            }
        } else if (mode === 'midline') {
            if (hoverPointIndex !== null && hoverPointIndex < annotations[currentFishIndex].midline_points.length) {
                const point = annotations[currentFishIndex].midline_points[hoverPointIndex];
                hoverX = dataToCanvasX(point.y);
                hoverY = dataToCanvasY(point.x);
            } else if (hoverOverHead) {
                const fish = annotations[currentFishIndex];
                hoverX = dataToCanvasX(fish.midline_points[0].y);
                hoverY = dataToCanvasY(fish.midline_points[0].x);
            } else {
                hoverX = lastMouseX;
                hoverY = lastMouseY;
            }
        }

        ctx.moveTo(hoverX, 0);
        ctx.lineTo(hoverX, canvas.height);
        ctx.moveTo(0, hoverY);
        ctx.lineTo(canvas.width, hoverY);
        ctx.stroke();
    }

    // Display information text
    const info = document.getElementById('info');
    info.innerText = `Mode: ${mode.toUpperCase()}`;
}

// Function to draw heads
function drawHeads() {
    annotations.forEach((fish, fishIndex) => {
        const headX = dataToCanvasX(fish.midline_points[0].y);
        const headY = dataToCanvasY(fish.midline_points[0].x);
        let fillStyle;
        // Change color if head is selected or being hovered
        if (fishIndex === selectedFish) {
            fillStyle = 'yellow';
        } else if (fishIndex === hoverFishIndex) {
            fillStyle = 'orange';
        } else {
            fillStyle = "rgba(255, 50, 50, .8)";
        }
        // Draw a plus sign
        drawPlusSign(headX, headY, 10, fillStyle);
    });
}

// Function to draw midlines
function drawMidlines() {
    if (annotations.length === 0) return;

    // Focus on the current fish
    const fish = annotations[currentFishIndex];
    if (!fish || !fish.midline_points || fish.midline_points.length === 0) return;

    // Zoom into the area around the fish head
    zoomToFishHead(fish);

    const midline = fish.midline_points;
    // Draw midline
    if (midline.length > 0) {
        ctx.beginPath();
        const firstPointX = dataToCanvasX(midline[0].y);
        const firstPointY = dataToCanvasY(midline[0].x);
        ctx.moveTo(firstPointX, firstPointY);
        midline.forEach((point) => {
            const canvasX = dataToCanvasX(point.y);
            const canvasY = dataToCanvasY(point.x);
            ctx.lineTo(canvasX, canvasY);
        });
        ctx.strokeStyle = 'red';
        ctx.stroke();

        // Draw points and text annotations
        midline.forEach((point, pointIndex) => {
            const canvasX = dataToCanvasX(point.y);
            const canvasY = dataToCanvasY(point.x);
            ctx.beginPath();
            ctx.arc(canvasX, canvasY, 5, 0, 2 * Math.PI);

            // Determine fill color based on visible_bool status
            if (!point.visible_bool) {
                ctx.fillStyle = 'rgba(255, 165, 0, 1)'; // Orange for visible
            } else {
                ctx.fillStyle = 'green'; // Green for occluded
            }

            // Highlight selected or hovered point
            if (pointIndex === selectedPointIndex) {
                ctx.strokeStyle = 'yellow';
                ctx.lineWidth = 2;
                ctx.stroke();
            } else if (pointIndex === hoverPointIndex) {
                ctx.strokeStyle = 'orange';
                ctx.lineWidth = 2;
                ctx.stroke();
            }

            ctx.fill();

            // Draw text annotation above the point
            ctx.fillStyle = "white";
            ctx.strokeStyle = "black";
            ctx.lineWidth = 3;
            ctx.font = '20px Arial';
            ctx.textAlign = 'center';
            ctx.strokeText(pointIndex + 1, canvasX + 10, canvasY - 10);
            ctx.fillText(pointIndex + 1, canvasX + 10, canvasY - 10);
        });
    }
    // Draw head position
    const headX = dataToCanvasX(fish.midline_points[0].y);
    const headY = dataToCanvasY(fish.midline_points[0].x);
    let headColor = "blue";

    if (hoverOverHead) {
        headColor = "orange"; // Change color on hover
    }

    drawPlusSign(headX, headY, 10, headColor);
}

// Function to zoom into fish head area
function zoomToFishHead(fish) {
    if (!fish || !fish.midline_points || fish.midline_points.length === 0) return;
    const zoomSize = 100; // Zoom area size in original image pixels

    // Calculate zoom area boundaries
    const headX = fish.midline_points[0].y;
    const headY = fish.midline_points[0].x;

    const zoomLeft = Math.max(headX - zoomSize / 2, 0);
    const zoomTop = Math.max(headY - zoomSize / 2, 0);
    const zoomWidth = Math.min(zoomSize, imgWidth - zoomLeft);
    const zoomHeight = Math.min(zoomSize, imgHeight - zoomTop);

    // Update scaling and offsets
    scale = Math.min(canvas.width / zoomWidth, canvas.height / zoomHeight);
    offsetX = -zoomLeft * scale;
    offsetY = -zoomTop * scale;
}

// Variables to track mouse position for hover feedback
let lastMouseX = 0;
let lastMouseY = 0;

// Event listeners for canvas interactions
canvas.addEventListener('mousedown', function (e) {
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    isDragging = false;

    // Convert mouse coordinates to data coordinates
    const dataX = canvasToDataY(mouseY); // Because in data x corresponds to rows (vertical)
    const dataY = canvasToDataX(mouseX); // In data y corresponds to columns (horizontal)

    if (mode === 'head') {
        selectedFish = null;
        selectedPointIndex = null;

        // Check if Shift key is pressed for adding a new head
        if (e.shiftKey) {
            saveState();
            annotations.push({
                case: "reliable", // Default value
                rolling_proba: 0.0, // Default value
                midline_points: [{x: dataX, y: dataY, visible_bool: true}]
            });
            selectedFish = annotations.length - 1;
            isDragging = false;
            drawAnnotations();
            return;
        }

        // Find the closest head
        let minDist = Infinity;
        annotations.forEach((fish, fishIndex) => {
            const dx = dataX - fish.midline_points[0].x;
            const dy = dataY - fish.midline_points[0].y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 10 && dist < minDist) {
                minDist = dist;
                selectedFish = fishIndex;
                isDragging = true;
            }
        });

        // If no head is close enough, do nothing
    } else if (mode === 'midline') {
        const fish = annotations[currentFishIndex];
        if (!fish || !fish.midline_points || fish.midline_points.length === 0) {
            drawAnnotations();
            return;
        }

        // Alt+Click to select head position for moving
        if (e.altKey) {
            const headCanvasX = dataToCanvasX(fish.midline_points[0].y);
            const headCanvasY = dataToCanvasY(fish.midline_points[0].x);
            const dx = mouseX - headCanvasX;
            const dy = mouseY - headCanvasY;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 10) { // Adjust threshold as needed
                selectedHead = true;
                isDragging = true;
                drawAnnotations();
                return;
            }
        }

        // Check if Shift key is pressed for adding a new point
        if (e.shiftKey) {
            saveState();
            fish.midline_points.push({x: dataX, y: dataY, visible_bool: true});
            selectedPointIndex = fish.midline_points.length - 1;
            isDragging = false;
            drawAnnotations();
            return;
        }

        selectedPointIndex = null;

        // Find the closest midline point
        let minDist = Infinity;
        fish.midline_points.forEach((point, idx) => {
            const dx = dataX - point.x;
            const dy = dataY - point.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 10 && dist < minDist) {
                minDist = dist;
                selectedPointIndex = idx;
                selectedFish = currentFishIndex;
                isDragging = true;
            }
        });

        // Select the point if found
        if (selectedPointIndex !== null) {
            selectedFish = currentFishIndex;
        } else {
            selectedFish = null;
        }
    }

    drawAnnotations();
});

canvas.addEventListener('mousemove', function (e) {
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    lastMouseX = mouseX;
    lastMouseY = mouseY;

    // Convert mouse coordinates to data coordinates
    const dataX = canvasToDataY(mouseY); // Because in data x corresponds to rows (vertical)
    const dataY = canvasToDataX(mouseX); // In data y corresponds to columns (horizontal)

    if (isDragging) {
        if (mode === 'head' && selectedFish !== null) {
            // Move head position
            annotations[selectedFish].midline_points[0].x = dataX;
            annotations[selectedFish].midline_points[0].y = dataY;
            drawAnnotations();
        } else if (mode === 'midline') {
            if (selectedHead) {
                // Move head position in midline mode
                annotations[currentFishIndex].midline_points[0].x = dataX;
                annotations[currentFishIndex].midline_points[0].y = dataY;
                drawAnnotations();
            } else if (selectedPointIndex !== null) {
                // Update the position of the selected point
                annotations[currentFishIndex].midline_points[selectedPointIndex].x = dataX;
                annotations[currentFishIndex].midline_points[selectedPointIndex].y = dataY;
                drawAnnotations();
            }
        }
    } else {
        // Handle hover feedback
        hoverFishIndex = null;
        hoverPointIndex = null;
        hoverOverHead = false;

        if (mode === 'head') {
            // Find the closest head
            let minDist = Infinity;
            annotations.forEach((fish, fishIndex) => {
                const dx = dataX - fish.midline_points[0].x;
                const dy = dataY - fish.midline_points[0].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 10 && dist < minDist) {
                    minDist = dist;
                    hoverFishIndex = fishIndex;
                }
            });
        } else if (mode === 'midline') {
            const fish = annotations[currentFishIndex];

            // Find the closest midline point
            let minDist = Infinity;
            fish.midline_points.forEach((point, idx) => {
                const dx = dataX - point.x;
                const dy = dataY - point.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 10 && dist < minDist) {
                    minDist = dist;
                    hoverPointIndex = idx;
                }
            });

            // Provide hover feedback for head position
            const headCanvasX = dataToCanvasX(fish.midline_points[0].y);
            const headCanvasY = dataToCanvasY(fish.midline_points[0].x);
            const dx = mouseX - headCanvasX;
            const dy = mouseY - headCanvasY;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 10) {
                hoverOverHead = true;
            }
        }
        drawAnnotations();
    }
});

canvas.addEventListener('mouseup', function (e) {
    if (isDragging) {
        if (selectedHead) saveState();
        isDragging = false;
        selectedHead = false;
        drawAnnotations();
    }
});

// Button event listeners
document.getElementById('modeHeadButton').addEventListener('click', function () {
    mode = 'head';
    selectedPointIndex = null;
    selectedFish = null;
    resetView();
    drawAnnotations();
    updateProgressBar(); // Hide progress bar in head mode
});

document.getElementById('modeMidlineButton').addEventListener('click', function () {
    mode = 'midline';
    currentFishIndex = 0;
    selectedPointIndex = null;
    selectedFish = null;
    drawAnnotations();
    updateProgressBar();
    // Trigger hover feedback
    canvas.dispatchEvent(new MouseEvent('mousemove', {clientX: lastMouseX, clientY: lastMouseY}));
});

// Add references to the progress bar elements
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');

function updateProgressBar() {
    if (mode === 'midline') {
        const totalFish = annotations.length;
        const currentFish = currentFishIndex + 1; // Zero-based index
        const progressPercentage = totalFish ? (currentFish / totalFish) * 100 : 0;
        progressFill.style.width = progressPercentage + '%';
        progressText.innerText = `Fish ${currentFish} of ${totalFish}`;
    } else {
        // Hide progress bar when not in midline mode
        progressFill.style.width = '0%';
        progressText.innerText = '';
    }
}

// Call updateProgressBar whenever the current fish changes
function navigateToFish(index) {
    currentFishIndex = index;
    selectedPointIndex = null;
    selectedFish = null;
    selectedHead = false;
    hoverOverHead = false;
    drawAnnotations();
    updateProgressBar();
    // Trigger hover feedback
    canvas.dispatchEvent(new MouseEvent('mousemove', {clientX: lastMouseX, clientY: lastMouseY}));
}

autoAnnotateButton = document.getElementById('autoAnnotateButton');
autoAnnotateButton.addEventListener('click', function () {
    if (mode === "midline") {
        saveState();

        // Same code as in the keydown event
        const fish = annotations[currentFishIndex];

        fetch("/auto_annotate_midline", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                experiment: experimentName,
                video: videoName,
                frame: frameName,
                head_pos: [fish.midline_points[0].y, fish.midline_points[0].x]
            })
        })
            .then(response => response.json())
            .then(data => {
                // Update fish.midline_points with predicted_points
                fish.midline_points = data.predicted_points;
                // Redraw the canvas
                drawAnnotations();
            })
            .catch(err => console.error("Auto-annotation error:", err));
    } else {
        alert("Auto-annotation is only available in midline mode.");
    }

});
// Keyboard events
document.addEventListener('keydown', function (e) {
    // Undo/Redo functionality
    if (e.key === "&") {  // Adjust if necessary, or use a button's click
        // Get currently selected fish head position from annotations or UI
        autoAnnotateButton.click();
    }

    if (e.key === '<') {
        isSkelBtnPressed = !isSkelBtnPressed;
        drawAnnotations();
    }


    if (e.ctrlKey && (e.key === 'z' || e.key === 'Z')) {
        if (e.shiftKey) {
            redo();
        } else {
            undo();
        }
        e.preventDefault();
        return;
    }

    if (mode === 'head') {
        if (selectedFish !== null) {
            const fish = annotations[selectedFish];
            if (e.key === 'Delete' || e.key === 'Backspace') {
                saveState();
                // Delete the selected head
                annotations.splice(selectedFish, 1);
                selectedFish = null;
                drawAnnotations();
            } else if (e.key === 'z') {
                fish.midline_points[0].x -= 0.5;
                drawAnnotations();

            } else if (e.key === 's') {
                fish.midline_points[0].x += 0.5;
                drawAnnotations();
            } else if (e.key === 'q') {
                fish.midline_points[0].y -= 0.5;
                drawAnnotations();
            } else if (e.key === 'd') {
                fish.midline_points[0].y += 0.5;
                drawAnnotations();
            }
        }
    } else if (mode === 'midline') {
        if (e.key === 'é') {
            saveState();
            const fish = annotations[currentFishIndex];
            const midlinePoints = fish.midline_points.map(point => [point.x, point.y]);
            const numberOfPoints = midlinePoints.length;
            const smoothingFactor = 10;

            fetch('/smooth_points', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    points: midlinePoints,
                    number_of_points: numberOfPoints,
                    smoothing_factor: smoothingFactor
                })
            })
                .then(response => response.json())
                .then(smoothedPoints => {
                    annotations[currentFishIndex].midline_points = smoothedPoints.map((point, index) => ({
                        x: point[0],
                        y: point[1],
                        visible_bool: fish.midline_points[index] ? fish.midline_points[index].visible_bool : true // preserve visible_bool or default to false
                    }));
                    drawAnnotations();

                })
                .catch(error => {
                    console.error('Error smoothing points:', error);
                });
        } else if (selectedPointIndex !== null) {
            const point = annotations[currentFishIndex].midline_points[selectedPointIndex];
            if (e.key === 'Delete' || e.key === 'Backspace') {
                saveState();
                // Delete the selected midline point
                annotations[currentFishIndex].midline_points.splice(selectedPointIndex, 1);
                selectedPointIndex = null;
                drawAnnotations();
            } else if (e.key === 'z') {
                point.x -= 0.5;
                drawAnnotations();
            } else if (e.key === 's') {
                point.x += 0.5;
                drawAnnotations();
            } else if (e.key === 'q') {
                point.y -= 0.5;
                drawAnnotations();
            } else if (e.key === 'd') {
                point.y += 0.5;
                drawAnnotations();
            } else if (e.key === 'a' && selectedPointIndex > 0) {
                saveState();
                swapPoints(annotations[currentFishIndex].midline_points, selectedPointIndex, selectedPointIndex - 1);
                selectedPointIndex--;
                drawAnnotations();
            } else if (e.key === 'e' && selectedPointIndex < annotations[currentFishIndex].midline_points.length - 1) {
                saveState();
                swapPoints(annotations[currentFishIndex].midline_points, selectedPointIndex, selectedPointIndex + 1);
                selectedPointIndex++;
                drawAnnotations();
            } else if (e.key === ' ') {
                // Toggle visible_bool with space bar
                saveState();
                point.visible_bool = !point.visible_bool;
                drawAnnotations();
                e.preventDefault(); // Prevent default space bar scrolling
            }

            canvas.dispatchEvent(new MouseEvent('mousemove', {clientX: lastMouseX, clientY: lastMouseY}));
        }

        // Handle Enter key to save and go to next fish or frame
        if (e.key === 'Enter') {
            if (currentFishIndex === annotations.length - 1) {
                // All fish reviewed, save and go to next frame
                saveAnnotationsAndNext();
            } else {
                // Navigate to next fish
                navigateToFish(currentFishIndex + 1);
            }
        }
    }

    // Mode switching
    if (e.key.toLowerCase() === 'h') {
        mode = 'head';
        selectedPointIndex = null;
        selectedFish = null;
        resetView();
        drawAnnotations();
        updateProgressBar(); // Hide progress bar in head mode
    } else if (e.key.toLowerCase() === 'm') {
        mode = 'midline';
        currentFishIndex = 0;
        selectedPointIndex = null;
        selectedFish = null;
        drawAnnotations();
        updateProgressBar();
        // Trigger hover feedback
        canvas.dispatchEvent(new MouseEvent('mousemove', {clientX: lastMouseX, clientY: lastMouseY}));
    }

    // Navigate between fish in midline mode
    if (mode === 'midline') {
        if (e.key === 'ArrowRight') {
            if (currentFishIndex < annotations.length - 1) {
                navigateToFish(currentFishIndex + 1);
            }
        } else if (e.key === 'ArrowLeft') {
            if (currentFishIndex > 0) {
                navigateToFish(currentFishIndex - 1);
            }
        }
    }
});

// Initialize progress bar on page load
window.addEventListener('load', function () {
    updateProgressBar();
});

// Handle "Save and Next" button click
nextBtn = document.getElementById('nextButton');
if (nextBtn) {
    nextBtn.addEventListener('click', function () {
        // Send annotations to the server
        fetch(reviewSaveUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(annotations)
        })
            .then(response => response.json())
            .then(data => {
                if (nextFrameUrl) {
                    // Redirect to next frame
                    window.location.href = nextFrameUrl;
                } else {
                    alert('All frames have been reviewed.');
                    // Optionally redirect back to the video or experiment page
                }
            })
            .catch(error => {
                console.error('Error saving annotations:', error);
                alert('Error saving annotations.');
            });
    });
}

nextRandomBtn = document.getElementById('nextRandomButton');

if (nextRandomBtn) {
    nextRandomBtn.addEventListener('click', function () {
        fetch(saveUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(annotations)
        })
            .then(response => response.json())
            .then(data => {
                window.location.href = nextRandomFrameUrl;
            })
            .catch(error => {
                console.error('Error saving annotations:', error);
                alert('Error saving annotations.');
            });
    });
}

nextReReviewButton = document.getElementById('nextReReviewButton');

if (nextReReviewButton) {
    nextReReviewButton.addEventListener('click', function () {
        fetch(saveUrl, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(annotations)
        })
            .then(res => res.json())
            .then(data => {
                // After saving, redirect to the next re-review frame
                window.location.href = nextReReviewFrameUrl;
            })
            .catch(err => {
                console.error("Error saving re-review:", err);
                alert('Error saving re-review.');
            });
    });
}


function saveAnnotationsAndNext() {
    // Send annotations to the server
    if (nextBtn) {
        nextBtn.click();
    } else if (nextRandomBtn) {
        nextRandomBtn.click();
    } else if (nextReReviewButton) {
        nextReReviewButton.click();
    }


}

// Undo/Redo functions
function saveState() {
    undoStack.push(JSON.stringify(annotations));
    redoStack = []; // Clear redo stack
}

function undo() {
    if (undoStack.length > 0) {
        redoStack.push(JSON.stringify(annotations));
        annotations = JSON.parse(undoStack.pop());
        drawAnnotations();
    }
}

function redo() {
    if (redoStack.length > 0) {
        undoStack.push(JSON.stringify(annotations));
        annotations = JSON.parse(redoStack.pop());
        drawAnnotations();
    }
}

// Function to reset view to full image
function resetView() {
    scale = 1;
    offsetX = 0;
    offsetY = 0;
    fitImageToCanvas();
}


// Button event listeners
document.getElementById('saveButton').addEventListener('click', function () {
    // Send annotations to the server
    fetch(saveUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(annotations)
    })
        .then(response => response.json())
        .then(data => {
            alert('Annotations saved successfully.');
        })
        .catch(error => {
            console.error('Error saving annotations:', error);
            alert('Error saving annotations.');
        });
});

document.getElementById('undoButton').addEventListener('click', function () {
    undo();
});

document.getElementById('redoButton').addEventListener('click', function () {
    redo();
});

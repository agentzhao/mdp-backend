import time
from algo.algo import MazeSolver
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import *
from helper import command_generator
from consts import Direction

import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image


app = Flask(__name__)
CORS(app)

model = YOLO(
    "Users/irfaa/OneDrive - Nanyang Technological University/Desktop/firstModel19/weights/best.pt"
)


@app.route("/")
def index():
    """
    Home route that displays hyperlinks to other routes
    """
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Home</title>
    </head>
    <body>
        <h1>Home</h1>
        <ul>
            <li><a href="/status">Status</a></li>
            <li><a href="/path">Path Finding (POST Request)</a></li>
            <li><a href="/image">Image Predict (POST Request)</a></li>
            <li><a href="/stitch">Stitch Images</a></li>
        </ul>
    </body>
    </html>
    """


@app.route("/status", methods=["GET"])
def status():
    """
    This is a health check endpoint to check if the server is running
    :return: a json object with a key "result" and value "ok"
    """
    return jsonify({"result": "ok"})


@app.route("/path", methods=["POST"])
def path_finding():
    """
    This is the main endpoint for the path finding algorithm
    :return: a json object with a key "data" and value a dictionary with keys "distance", "path", and "commands"

    Example JSON Request with 2 obstacles:
    {"obstacles":[{"x":6,"y":8,"d":0,"id":9},{"x":16,"y":18,"d":6,"id":8}],"robot_x":1,"robot_y":1,"robot_dir":0,"retrying":false}
    """
    # Get the json data from the request
    content = request.json

    assert content

    # Get the obstacles, big_turn, retrying, robot_x, robot_y, and robot_direction from the json data
    obstacles = content["obstacles"]
    # big_turn = int(content['big_turn'])
    retrying = content["retrying"]
    robot_x, robot_y = content["robot_x"], content["robot_y"]
    robot_direction = int(content["robot_dir"])

    # Initialize MazeSolver object with robot size of 20x20, bottom left corner of robot at (1,1), facing north, and whether to use a big turn or not.
    maze_solver = MazeSolver(
        size_x=20,
        size_y=20,
        robot_x=robot_x,
        robot_y=robot_y,
        robot_direction=Direction(robot_direction),
        big_turn=0,
    )

    # Add each obstacle into the MazeSolver. Each obstacle is defined by its x,y positions, its direction, and its id
    for ob in obstacles:
        maze_solver.add_obstacle(ob["x"], ob["y"], ob["d"], ob["id"])

    start = time.time()
    # Get shortest path
    optimal_path, distance = maze_solver.get_optimal_order_dp(retrying=retrying)
    print(f"Time taken to find shortest path using A* search: {time.time() - start}s")
    print(f"Distance to travel: {distance} units")

    # Based on the shortest path, generate commands for the robot
    commands = command_generator(optimal_path, obstacles)

    # Get the starting location and add it to path_results
    path_results = [optimal_path[0].get_dict()]
    # Process each command individually and append the location the robot should be after executing that command to path_results
    i = 0
    for command in commands:
        if command.startswith("SNAP"):
            continue
        if command.startswith("FIN"):
            continue
        elif command.startswith("FW") or command.startswith("FS"):
            i += int(command[2:]) // 10
        elif command.startswith("BW") or command.startswith("BS"):
            i += int(command[2:]) // 10
        else:
            i += 1
        path_results.append(optimal_path[i].get_dict())
    return jsonify(
        {
            "data": {"distance": distance, "path": path_results, "commands": commands},
            "error": None,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    This is the main endpoint for the image prediction algorithm
    :return: a json object with a key "result" and value a dictionary with keys "obstacle_id" and "image_id"
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # Read the file as an image
        img = Image.open(file.stream)

        # Convert the image to a NumPy array
        img = np.array(img)

        # Convert RGB to BGR format (YOLO expects BGR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Run YOLOv8 inference
        results = model(img)  # This returns a list of Results objects
        result = results[0]  # Access the first Results object

        # Extract bounding boxes and labels from the results
        boxes = result.boxes  # Bounding boxes
        names = result.names  # Class names

        # Prepare predictions in a list of dictionaries
        predictions = []
        for box in boxes:
            predictions.append(
                {
                    "x1": box.xyxy[0][0].item(),
                    "y1": box.xyxy[0][1].item(),
                    "x2": box.xyxy[0][2].item(),
                    "y2": box.xyxy[0][3].item(),
                    "confidence": box.conf[0].item(),
                    "class_id": box.cls[0].item(),
                    "class_name": names[int(box.cls[0].item())],
                }
            )

        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# idk what is this for
@app.route("/stitch", methods=["GET"])
def stitch():
    """
    This is the main endpoint for the stitching command. Stitches the images using two different functions, in effect creating two stitches, just for redundancy purposes
    """
    img = stitch_image()
    img.show()
    img2 = stitch_image_own()
    img2.show()
    return jsonify({"result": "ok"})


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=5000, debug=True)
    app.run(port=5000, debug=True)

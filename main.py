import time
import logging
import os
from algo.algo import MazeSolver
from flask import Flask, request, jsonify
from flask_cors import CORS
from helper import command_generator
from consts import Direction
import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image

# from model import *

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)
model = YOLO("./best.pt")


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
    # Based on the shortest path, generate commands for the robot
    commands = command_generator(optimal_path, obstacles)

    print(f"Time taken to find shortest path using A* search: {time.time() - start}s")
    print(f"Distance to travel: {distance} units")
    print(f"Path: {optimal_path}")
    print(f"Commands: {commands}")

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


@app.route("/image", methods=["POST"])
def image_predict():
    """
    This is the main endpoint for the image prediction algorithm
    :return: a json object with a key "result" and value a dictionary with keys "obstacle_id" and "image_id"
    """
    logging.debug("Received image prediction request")
    if "file" not in request.files:
        logging.error("No file part in the request")
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        logging.error("No file selected")
        return jsonify({"error": "No file selected"}), 400

    if file:
        filename = file.filename
        logging.debug(f"Received file: {filename}")

        upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        file_path = os.path.join(upload_dir, filename)

        try:
            file.save(file_path)
            logging.debug(f"File saved successfully: {file_path}")
        except Exception as e:
            logging.error(f"Failed to save file: {str(e)}")
            return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

        obstacle_id = os.path.splitext(filename)[0]

        try:
            image_id, annotated_image = predict_image_week_9(filename, model)
            logging.debug(f"Prediction successful. Image ID: {image_id}")

            # Save the annotated image
            prediction_dir = os.path.join(os.path.dirname(__file__), "predictions")
            if not os.path.exists(prediction_dir):
                os.makedirs(prediction_dir)
            annotated_file_path = os.path.join(prediction_dir, f"annotated_{filename}")
            cv2.imwrite(annotated_file_path, annotated_image)
            logging.debug(f"Annotated image saved: {annotated_file_path}")
        except Exception as e:
            logging.error(f"Error in predict_image_week_9: {str(e)}")
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

        result = {
            # "obstacle_id": obstacle_id,
            "image_id": image_id
        }
        # logging.debug(f"Returning result: {result}")
        return jsonify(result)

    logging.error("Unknown error occurred")
    return jsonify({"error": "Unknown error occurred"}), 500


def predict_image_week_9(filename, model):
    # Read the image
    img = cv2.imread(os.path.join("uploads", filename))

    # Perform inference
    results = model(img)

    # Process results
    detections = results[0].boxes.data  # Get detection data

    # Extract class IDs of detected objects
    class_ids = detections[:, 5].cpu().numpy().astype(int)

    # Define the name_to_id dictionary
    name_to_id = {
        "NA": "NA",
        "BullsEye": 11,
        "1": 0,
        "2": 1,
        "3": 2,
        "4": 3,
        "5": 4,
        "6": 5,
        "7": 6,
        "8": 7,
        "9": 8,
        "A": 9,
        "B": 10,
        "C": 12,
        "D": 13,
        "E": 16,
        "F": 17,
        "G": 18,
        "H": 19,
        "S": 22,
        "T": 23,
        "U": 24,
        "V": 26,
        "W": 27,
        "X": 28,
        "Y": 29,
        "Z": 30,
        "Up": 25,
        "Down": 15,
        "Right": 21,
        "Left": 20,
        "Dot": 14,
    }

    # Create a reverse mapping
    id_to_name = {v: k for k, v in name_to_id.items() if v != "NA"}

    # Get the most common class ID (excluding background class if applicable)
    if len(class_ids) > 0:
        most_common_id = np.bincount(class_ids).argmax()
        image_id = id_to_name.get(most_common_id, "NA")
    else:
        image_id = "NA"  # No detection

    # Annotate the image
    for det in detections:
        bbox = det[:4].cpu().numpy().astype(int)
        conf = det[4].cpu().numpy()
        class_id = int(det[5].cpu().numpy())

        # Get the class name
        class_name = id_to_name.get(class_id, "Unknown")

        # Draw bounding box
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Add label
        label = f"{class_name}: {conf:.2f}"
        cv2.putText(
            img,
            label,
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Return the image id and annotated image
    return image_id, img.copy()  # Return a copy of the numpy array


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

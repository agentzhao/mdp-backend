import time
import re
from algo.algo import MazeSolver
from flask import Flask, request, jsonify
from flask_cors import CORS
from helper import command_generator
from consts import Direction
from ultralytics import YOLO
import cv2
import numpy as np
import os
import logging

logging.basicConfig(level=logging.DEBUG)


app = Flask(__name__)
CORS(app)

model = YOLO("./best_wk6.pt")


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
    {"obstacles":[{"x":6,"y":8,"d":0,"id":9},{"x":16,"y":18,"d":6,"id":8}],"robot_x":1,"robot_y":1,"robot_dir":0}
    """
    # Get the json data from the request
    content = request.json

    assert content

    # Get the obstacles, big_turn, retrying, robot_x, robot_y, and robot_direction from the json data
    try:
        # Extract and validate obstacles
        obstacles = content["obstacles"]
        if not isinstance(obstacles, list):
            raise ValueError("Obstacles should be a list of objects")

        for obstacle in obstacles:
            # Ensure each obstacle has the expected keys
            if not all(key in obstacle for key in ("x", "y", "d", "id")):
                raise ValueError(
                    "Each obstacle must have 'x', 'y', 'd', and 'id' properties"
                )

        # Extract and validate retrying
        # retrying = content["retrying"]
        # if not isinstance(retrying, bool):
        #     raise ValueError("'retrying' should be a boolean")
        #
        # # Extract and validate robot_x and robot_y
        # robot_x = content["robot_x"]
        # robot_y = content["robot_y"]
        # if not (isinstance(robot_x, int) and isinstance(robot_y, int)):
        #     raise ValueError("'robot_x' and 'robot_y' should be numbers")
        #
        # # Extract and validate robot_direction
        # robot_direction = int(content["robot_dir"])
        # if not isinstance(robot_direction, int):
        #     raise ValueError("'robot_dir' should be an integer")

    except KeyError as e:
        return jsonify({"error": f"Missing key: {str(e)}"}), 400

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    # big_turn = int(content['big_turn'])

    # hardcoded values
    robot_x = 1
    robot_y = 1
    robot_direction = 0
    retrying = False
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
    commands = updateCommands(commands)

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
            # rpi take picture
            print("Image captured and sent for processing")
            continue
        if command.startswith("FIN"):
            continue
        elif command.startswith("FW") or command.startswith("FS"):
            i += int(command[2:]) // 10
        elif command.startswith("BW") or command.startswith("BS"):
            i += int(command[2:]) // 10
        else:
            ##send_command_to_stm(command)
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

        try:
            image_id, image_name, annotated_image = predict_image_week_9(
                filename, model
            )
            logging.debug(
                f"Prediction successful. Image ID: {image_id}. Image Name: {image_name}"
            )

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
            "image_id": image_id,
            "image_name": image_name,
        }
        # logging.debug(f"Returning result: {result}")
        return jsonify(result)

    logging.error("Unknown error occurred")
    return jsonify({"error": "Unknown error occurred"}), 500


### Helper functions
def updateCommands(commands):
    # Combine FW and BW commands if consecutive
    combined_commands = []
    i = 0
    while i < len(commands):
        if commands[i].startswith("FW") or commands[i].startswith("BW"):
            # Start adding up consecutive FW or BW values
            cmd_type = commands[i][:2]  # Either "FW" or "BW"
            cmd_sum = int(re.findall(r"\d+", commands[i])[0])
            while i + 1 < len(commands) and commands[i + 1].startswith(cmd_type):
                cmd_sum += int(re.findall(r"\d+", commands[i + 1])[0])
                i += 1
            combined_commands.append(f"{cmd_type}{cmd_sum:03d}")
        else:
            combined_commands.append(commands[i])
        i += 1

    ## change the FR BR turning commands to 090
    updated_commands = []
    for command in combined_commands:
        if command in ["FR00", "FL00", "BR00", "BL00"]:
            updated_commands.append(command[:2] + "090")
        else:
            updated_commands.append(command)

    return updated_commands


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
    name_to_yolo_id = {
        "NA": "NA",
        "Bullseye": 11,
        "One": 0,
        "Two": 1,
        "Three": 2,
        "Four": 3,
        "Five": 4,
        "Six": 5,
        "Seven": 6,
        "Eight": 7,
        "Nine": 8,
        "Alphabet A": 9,
        "Alphabet B": 10,
        "Alphabet C": 12,
        "Alphabet D": 13,
        "Alphabet E": 16,
        "Alphabet F": 17,
        "Alphabet G": 18,
        "Alphabet H": 19,
        "Alphabet S": 22,
        "Alphabet T": 23,
        "Alphabet U": 24,
        "Alphabet V": 26,
        "Alphabet W": 27,
        "Alphabet X": 28,
        "Alphabet Y": 29,
        "Alphabet Z": 30,
        "Up arrow": 25,
        "Down arrow": 15,
        "Right arrow": 21,
        "Left arrow": 20,
        "Stop": 14,
    }

    name_to_image_id = {
        "NA": "NA",
        "Bullseye": 10,
        "One": 11,
        "Two": 12,
        "Three": 13,
        "Four": 14,
        "Five": 15,
        "Six": 16,
        "Seven": 17,
        "Eight": 18,
        "Nine": 19,
        "Alphabet A": 20,
        "Alphabet B": 21,
        "Alphabet C": 22,
        "Alphabet D": 23,
        "Alphabet E": 24,
        "Alphabet F": 25,
        "Alphabet G": 26,
        "Alphabet H": 27,
        "Alphabet S": 28,
        "Alphabet T": 29,
        "Alphabet U": 30,
        "Alphabet V": 31,
        "Alphabet W": 32,
        "Alphabet X": 33,
        "Alphabet Y": 34,
        "Alphabet Z": 35,
        "Up arrow": 36,
        "Down arrow": 37,
        "Right arrow": 38,
        "Left arrow": 39,
        "Stop": 40,
        "Bullseye": 41,
    }

    # Create a reverse mapping
    id_to_name = {v: k for k, v in name_to_yolo_id.items() if v != "NA"}

    # Get the most common class ID (excluding background class if applicable)
    if len(class_ids) > 0:
        most_common_id = np.bincount(class_ids).argmax()
        image_name = id_to_name.get(most_common_id, "NA")
    else:
        image_name = "NA"  # No detection

    # Get the image ID based on the image name using the name_to_image_id dictionary
    image_id = name_to_image_id.get(image_name, "NA")

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
    return image_id, image_name, img.copy()  # Return a copy of the numpy array


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=5000, debug=True)
    app.run(port=5000, debug=True)

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
from PIL import Image
import glob
import shutil

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
    {"obstacles":[{"x":6,"y":8,"d":0,"id":9},{"x":16,"y":18,"d":6,"id":8}],"robot_x":1,"robot_y":1,"robot_dir":0,"retrying":False}
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
        retrying = content["retrying"]
        if not isinstance(retrying, bool):
            raise ValueError("'retrying' should be a boolean")

        # Extract and validate robot_x and robot_y
        robot_x = content["robot_x"]
        robot_y = content["robot_y"]
        if not (isinstance(robot_x, int) and isinstance(robot_y, int)):
            raise ValueError("'robot_x' and 'robot_y' should be numbers")

        # Extract and validate robot_direction
        robot_direction = int(content["robot_dir"])
        if not isinstance(robot_direction, int):
            raise ValueError("'robot_dir' should be an integer")

    except KeyError as e:
        return jsonify({"error": f"Missing key: {str(e)}"}), 400

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    # big_turn = int(content['big_turn'])

    # hardcoded values
    # robot_x = 1
    # robot_y = 1
    # robot_direction = 0
    # retrying = False
    big_turn = 0
    # Initialize MazeSolver object with robot size of 20x20, bottom left corner of robot at (1,1), facing north, and whether to use a big turn or not.
    maze_solver = MazeSolver(
        size_x=20,
        size_y=20,
        robot_x=robot_x,
        robot_y=robot_y,
        robot_direction=Direction(robot_direction),
        big_turn=big_turn,
    )

    # Add each obstacle into the MazeSolver. Each obstacle is defined by its x,y positions, its direction, and its id
    for ob in obstacles:
        maze_solver.add_obstacle(ob["x"], ob["y"], ob["d"], ob["id"])

    start = time.time()
    # Get shortest path
    optimal_path, distance = maze_solver.get_optimal_order_dp(retrying=retrying)
    # Based on the shortest path, generate commands for the robot
    commands = command_generator(optimal_path, obstacles)
    updated_commands = updateCommands(commands)

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
            "data": {
                "distance": distance,
                "path": path_results,
                "commands": updated_commands,
            },
            "error": None,
        }
    )




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

    i = 0
    while i < len(commands):
        if commands[i].startswith("FW") or commands[i].startswith("BW"):
            # Extract the number and pad it with zeros to ensure it has 3 digits
            cmd_type = commands[i][:2]
            cmd_value = int(re.findall(r"\d+", commands[i])[0])
            combined_commands.append(f"{cmd_type}{cmd_value:03d}")
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


@app.route('/image', methods=['POST'])
def image_predict():
    logging.debug("Received image prediction request")
    if 'file' not in request.files:
        logging.error("No file part in the request")
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        logging.error("No file selected")
        return jsonify({"error": "No file selected"}), 400

    if file:
        filename = file.filename
        logging.debug(f"Received file: {filename}")
        
        upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        file_path = os.path.join(upload_dir, filename)
        
        try:
            file.save(file_path)
            logging.debug(f"File saved successfully: {file_path}")
        except Exception as e:
            logging.error(f"Failed to save file: {str(e)}")
            return jsonify({"error": f"Failed to save file: {str(e)}"}), 500
        
        # filename format: "<timestamp>_<obstacle_id>.jpeg"
        constituents = file.filename.split("_")
        obstacle_id = constituents[1].strip(".jpg")

        try:
            image_id, annotated_image = predict_image(filename, model)           
            logging.debug(f"Prediction successful. Image ID: {image_id}. Obstacle ID: {obstacle_id}")
            
            # Save the annotated image
            prediction_dir = os.path.join(os.path.dirname(__file__), 'predictions')
            if not os.path.exists(prediction_dir):
                os.makedirs(prediction_dir)
            annotated_file_path = os.path.join(prediction_dir, f"annotated_{filename}")
            cv2.imwrite(annotated_file_path, annotated_image)
            logging.debug(f"Annotated image saved: {annotated_file_path}")
        except Exception as e:
            logging.error(f"Error in image_predict: {str(e)}")
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

        result = {
            "obstacle_id": obstacle_id,
            "image_id":image_id
        }
        logging.debug(f"Returning result: {result}")
        
        
        return jsonify(result)

    logging.error("Unknown error occurred")
    return jsonify({"error": "Unknown error occurred"}), 500

@app.route('/stitch', methods=['GET'])
def stitch():
    """
    This is the main endpoint for the stitching command. 
    """
    img = stitch_image()
    img.show()
    # img2 = stitch_image_own()
    # img2.show()
    return jsonify({"result": "ok"})

def predict_image(filename, model): 
    # Read the image
    img = cv2.imread(os.path.join('uploads', filename))
    
    # Convert the image to a NumPy array 
    img = np.array(img) 
        
    # Convert RGB to BGR format (YOLO expects BGR) 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
        
    # Run YOLOv8 inference 
    results = model(img)  # This returns a list of Results objects 

    result = results[0]   # Access the first Results object 
        
    # Extract bounding boxes and labels from the results 
    boxes = result.boxes  # Bounding boxes 
    names = result.names  # Class names 

    name_to_id = {
        "NA": 'NA',
        "Bullseye": 10,
        "1": 11,
        "2": 12,
        "3": 13,
        "4": 14,
        "5": 15,
        "6": 16,
        "7": 17,
        "8": 18,
        "9": 19,
        "A": 20,
        "B": 21,
        "C": 22,
        "D": 23,
        "E": 24,
        "F": 25,
        "G": 26,
        "H": 27,
        "S": 28,
        "T": 29,
        "U": 30,
        "V": 31,
        "W": 32,
        "X": 33,
        "Y": 34,
        "Z": 35,
        "Up": 36,
        "Down": 37,
        "Right": 38,
        "Left": 39,
        "Dot": 40,
    }
    
    # Prepare predictions in a list of dictionaries 
    predictions_short = [] 
    for box in boxes: 
        class_name = names[int(box.cls[0].item())] 
        boxHt = box.xyxy[0][3].item() - box.xyxy[0][1].item() 
        boxWt = box.xyxy[0][2].item() -  box.xyxy[0][0].item() 
        boxArea = boxHt * boxWt 
        predictions_short.append({ 
            "x1": box.xyxy[0][0].item(), 
            "y1": box.xyxy[0][1].item(), 
            "x2": box.xyxy[0][2].item(), 
            "y2": box.xyxy[0][3].item(), 
            "confidence": box.conf[0].item(), 
            "class_id":  name_to_id.get(class_name, 'NA'), 
            "class_name": class_name, 
            "box_area" : boxArea 
        }) 
        
    if len(predictions_short) > 1: 
        logging.debug(f"More than one prediction found: {predictions_short}")
        
        # Filter out Bullseye predictions
        non_bullseye_predictions = [pred for pred in predictions_short if pred.get('class_name') != 'Bullseye']
        
        if non_bullseye_predictions:
            # Sort non-Bullseye predictions by the largest box area
            non_bullseye_predictions.sort(key=lambda x: x.get('box_area'), reverse=True)
            predictions = [non_bullseye_predictions[0]]  # Select the prediction with the largest area
            logging.debug(f"Prediction with largest box area (non-Bullseye): {predictions}")
        else:
            # If all predictions were Bullseye, sort Bullseye by the largest box area
            logging.debug(f"All predictions are Bullseye, using the largest Bullseye box area")
            predictions_short.sort(key=lambda x: x.get('box_area'), reverse=True)
            predictions = [predictions_short[0]]  # Select the largest Bullseye box
            logging.debug(f"Prediction with largest Bullseye box area: {predictions}")
    elif len(predictions_short) == 0: 
        predictions = [] 
        predictions.append({ 
            "x1": 0, 
            "y1": 0, 
            "x2": 0, 
            "y2": 0, 
            "confidence": 0, 
            "class_id":  'NA', 
            "class_name": 'NA', 
            "box_area" : 0 
        }) 
    else: 
        predictions = [predictions_short[0]] 
        
    image_id = predictions[0]["class_id"] 
        
    # Annotate the image    
    # Convert the coordinates to int
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bbox = [int(predictions[0]['x1']), int(predictions[0]['y1']), int(predictions[0]['x2']), int(predictions[0]['y2'])]
    conf = predictions[0]['confidence']
    image_name = predictions[0]["class_name"]
    
    # Draw bounding box
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    
    # Add label
    label = f"{image_name}: {conf:.2f}" 
    cv2.putText(img, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    return image_id, img.copy()

def stitch_image():
    """
    Stitches the images in the folder together and saves it into runs/stitched folder
    """
    # Initialize path to save stitched image
    imgFolder = 'predictions'
    stitchedPath = os.path.join(imgFolder, f'stitched-{int(time.time())}.jpeg')

    # Find all files that ends with ".jpg" (this won't match the stitched images as we name them ".jpeg")
    # imgPaths = glob.glob(os.path.join(imgFolder+"/detect/*/", "*.jpg"))
    imgPaths = glob.glob(os.path.join(imgFolder, "*.jpg"))
    # Open all images
    images = [Image.open(x) for x in imgPaths]
    # Get the width and height of each image
    width, height = zip(*(i.size for i in images))
    # Calculate the total width and max height of the stitched image, as we are stitching horizontally
    total_width = sum(width)
    max_height = max(height)
    stitchedImg = Image.new('RGB', (total_width, max_height))
    x_offset = 0

    # Stitch the images together
    for im in images:
        stitchedImg.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    # Save the stitched image to the path
    stitchedImg.save(stitchedPath)

    # Move original images to "originals" subdirectory
    original_dir = os.path.join(os.path.dirname(__file__), 'predictions/originals')
    if not os.path.exists(original_dir):
        os.makedirs(original_dir)
    for img in imgPaths:
        shutil.move(img, os.path.join(
            "predictions", "originals", os.path.basename(img)))

    return stitchedImg


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=5000, debug=True)
    app.run(port=5000, debug=True)

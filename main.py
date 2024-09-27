import time
import re
from algo.algo import MazeSolver
from flask import Flask, request, jsonify
from flask_cors import CORS
from helper import command_generator
from consts import Direction
from flask import Flask, request, jsonify

import requests
import io
from picamera import PiCamera
import serial

app = Flask(__name__)
CORS(app)

# STM32 Serial Configuration
STM_SERIAL_PORT = "/dev/ttyUSB0"
STM_BAUD_RATE = 115200
ser = serial.Serial(STM_SERIAL_PORT, STM_BAUD_RATE, timeout=1)

# Inference Server URL
INFERENCE_SERVER_URL = "http://192.168.51.77:5002/image"
image_counter = 0  # Global counter for image naming


def send_command_to_stm(command: str):
    """Send a command to the STM32 via serial and wait for response."""
    ser.write(f"{command}\n".encode("ascii"))
    print(f"Sent: {command}")
    time.sleep(7)  # Wait for STM32 to process the command
    if ser.in_waiting > 0:
        response = ser.readline().decode("utf-8").strip()
        print(f"Received: {response}")
    else:
        print("No response received from STM32")


def capture_and_send_image():
    """Capture an image and send it to the inference server."""
    global image_counter
    camera = PiCamera()
    camera.resolution = (1920, 1080)
    time.sleep(2)  # Camera warm-up time

    stream = io.BytesIO()
    camera.capture(stream, format="jpeg")
    stream.seek(0)

    image_counter += 1
    filename = f"image_{image_counter}.jpg"

    result = send_image_to_server(stream, filename)
    camera.close()
    return result


def send_image_to_server(image_stream, filename):
    """Send the captured image to the inference server."""
    files = {"file": (filename, image_stream, "image/jpeg")}
    try:
        response = requests.post(INFERENCE_SERVER_URL, files=files)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: Server responded with status code {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error sending request to server: {e}")
        return None


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
            result = capture_and_send_image()
            print("Image captured and sent for processing")
            continue
        if command.startswith("FIN"):
            continue
        elif command.startswith("FW") or command.startswith("FS"):
            i += int(command[2:]) // 10
        elif command.startswith("BW") or command.startswith("BS"):
            i += int(command[2:]) // 10
        else:
            send_command_to_stm(command)
            i += 1
        path_results.append(optimal_path[i].get_dict())
    return jsonify(
        {
            "data": {"distance": distance, "path": path_results, "commands": commands},
            "error": None,
        }
    )


@app.route("/test", methods=["GET"])
def test():
    commands = ["FW010", "FR090", "FW010", "FR090", "FW010", "FR090", "FW010", "FR090"]
    for command in commands:
        if command.startswith("SNAP"):
            # rpi take picture
            result = capture_and_send_image()
            print("Image captured and sent for processing")
            continue
        if command.startswith("FIN"):
            continue
        send_command_to_stm(command)
    return jsonify({"result": "ok"})


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=5000, debug=True)
    app.run(port=5000, debug=True)

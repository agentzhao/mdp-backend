import time
from algo.algo import MazeSolver
from flask import Flask, request, jsonify
from flask_cors import CORS
from helper import command_generator
from consts import Direction
from flask import Flask, request, jsonify
from PIL import Image
from model import *


app = Flask(__name__)
CORS(app)
# model = YOLO("./best.pt")


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
            "data": {"distance": distance, "path": path_results, "commands": commands},
            "error": None,
        }
    )


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

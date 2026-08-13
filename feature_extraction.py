import os
import csv
import math

import cv2
import mediapipe as mp


# ============================================================
# SETTINGS
# ============================================================

VIDEO_SOURCE = "data/raw/video.mp4"

OUTPUT_CSV = (
    "data/processed/features/pose_features.csv"
)

OUTPUT_VIDEO = (
    "data/processed/features/annotated_pose.mp4"
)

# Set True if you want to see the video while processing.
SHOW_VIDEO = True

# Set True if you want an annotated output video.
SAVE_ANNOTATED_VIDEO = True

# MediaPipe settings
MODEL_COMPLEXITY = 1

MIN_DETECTION_CONFIDENCE = 0.5

MIN_TRACKING_CONFIDENCE = 0.5


# ============================================================
# MEDIAPIPE
# ============================================================

mp_pose = mp.solutions.pose

mp_drawing = mp.solutions.drawing_utils


# ============================================================
# TRACKED LANDMARKS
# ============================================================

"""
MediaPipe Pose landmark IDs:

11 -> Left Shoulder
12 -> Right Shoulder
13 -> Left Elbow
14 -> Right Elbow
15 -> Left Wrist
16 -> Right Wrist
23 -> Left Hip
24 -> Right Hip
25 -> Left Knee
26 -> Right Knee
27 -> Left Ankle
28 -> Right Ankle
"""

TRACKED_LANDMARKS = {
    11: "LS",
    12: "RS",
    13: "LE",
    14: "RE",
    15: "LW",
    16: "RW",
    23: "LH",
    24: "RH",
    25: "LK",
    26: "RK",
    27: "LA",
    28: "RA",
}


# Keep a fixed order everywhere.
LANDMARK_ORDER = [
    11, 12, 13, 14, 15, 16,
    23, 24, 25, 26, 27, 28
]


# ============================================================
# ANGLE ORDER
# ============================================================

ANGLE_ORDER = [
    "Left Elbow",
    "Right Elbow",

    "Left Shoulder",
    "Right Shoulder",

    "Left Hip",
    "Right Hip",

    "Left Knee",
    "Right Knee",
]


# ============================================================
# FEATURE NAMES
# ============================================================

feature_names = [

    # --------------------------------------------------------
    # 36 COORDINATES
    # --------------------------------------------------------

    "LS_X", "LS_Y", "LS_Z",
    "RS_X", "RS_Y", "RS_Z",

    "LE_X", "LE_Y", "LE_Z",
    "RE_X", "RE_Y", "RE_Z",

    "LW_X", "LW_Y", "LW_Z",
    "RW_X", "RW_Y", "RW_Z",

    "LH_X", "LH_Y", "LH_Z",
    "RH_X", "RH_Y", "RH_Z",

    "LK_X", "LK_Y", "LK_Z",
    "RK_X", "RK_Y", "RK_Z",

    "LA_X", "LA_Y", "LA_Z",
    "RA_X", "RA_Y", "RA_Z",


    # --------------------------------------------------------
    # 10 BODY DISTANCES
    # --------------------------------------------------------

    "Shoulder_Width",
    "Hip_Width",

    "Left_Upper_Arm_Length",
    "Right_Upper_Arm_Length",

    "Left_Forearm_Length",
    "Right_Forearm_Length",

    "Left_Thigh_Length",
    "Right_Thigh_Length",

    "Left_Shin_Length",
    "Right_Shin_Length",


    # --------------------------------------------------------
    # 8 JOINT ANGLES
    # --------------------------------------------------------

    "Left_Elbow_Angle",
    "Right_Elbow_Angle",

    "Left_Shoulder_Angle",
    "Right_Shoulder_Angle",

    "Left_Hip_Angle",
    "Right_Hip_Angle",

    "Left_Knee_Angle",
    "Right_Knee_Angle",


    # --------------------------------------------------------
    # 12 LINEAR VELOCITIES
    # --------------------------------------------------------

    "LS_Velocity",
    "RS_Velocity",

    "LE_Velocity",
    "RE_Velocity",

    "LW_Velocity",
    "RW_Velocity",

    "LH_Velocity",
    "RH_Velocity",

    "LK_Velocity",
    "RK_Velocity",

    "LA_Velocity",
    "RA_Velocity",


    # --------------------------------------------------------
    # 12 LINEAR ACCELERATIONS
    # --------------------------------------------------------

    "LS_Acceleration",
    "RS_Acceleration",

    "LE_Acceleration",
    "RE_Acceleration",

    "LW_Acceleration",
    "RW_Acceleration",

    "LH_Acceleration",
    "RH_Acceleration",

    "LK_Acceleration",
    "RK_Acceleration",

    "LA_Acceleration",
    "RA_Acceleration",


    # --------------------------------------------------------
    # 8 ANGULAR VELOCITIES
    # --------------------------------------------------------

    "Left_Elbow_Angular_Velocity",
    "Right_Elbow_Angular_Velocity",

    "Left_Shoulder_Angular_Velocity",
    "Right_Shoulder_Angular_Velocity",

    "Left_Hip_Angular_Velocity",
    "Right_Hip_Angular_Velocity",

    "Left_Knee_Angular_Velocity",
    "Right_Knee_Angular_Velocity",


    # --------------------------------------------------------
    # 8 ANGULAR ACCELERATIONS
    # --------------------------------------------------------

    "Left_Elbow_Angular_Acceleration",
    "Right_Elbow_Angular_Acceleration",

    "Left_Shoulder_Angular_Acceleration",
    "Right_Shoulder_Angular_Acceleration",

    "Left_Hip_Angular_Acceleration",
    "Right_Hip_Angular_Acceleration",

    "Left_Knee_Angular_Acceleration",
    "Right_Knee_Angular_Acceleration",
]


# ============================================================
# FEATURE COUNT CHECK
# ============================================================

EXPECTED_FEATURE_COUNT = 94

print(
    "Total features:",
    len(feature_names)
)

if len(feature_names) != EXPECTED_FEATURE_COUNT:

    raise ValueError(
        f"Expected {EXPECTED_FEATURE_COUNT} features, "
        f"but got {len(feature_names)}"
    )


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def is_valid_point(point):
    """
    Check whether a 3D point contains valid values.
    """

    if point is None:
        return False

    return all(
        math.isfinite(value)
        for value in point
    )


# ------------------------------------------------------------


def distance(point_a, point_b):
    """
    Calculate Euclidean distance between two 3D points.

    d = sqrt(dx² + dy² + dz²)
    """

    if not is_valid_point(point_a):
        return math.nan

    if not is_valid_point(point_b):
        return math.nan

    dx = point_a[0] - point_b[0]

    dy = point_a[1] - point_b[1]

    dz = point_a[2] - point_b[2]

    return math.sqrt(
        dx * dx +
        dy * dy +
        dz * dz
    )


# ------------------------------------------------------------


def calculate_angle(point_a, point_b, point_c):
    """
    Calculate 3D angle ABC.

    The angle is measured at point B.

          A
           \
            B
           /
          C

    Returns:
        angle in degrees
    """

    if not is_valid_point(point_a):
        return math.nan

    if not is_valid_point(point_b):
        return math.nan

    if not is_valid_point(point_c):
        return math.nan

    vector_ba = (
        point_a[0] - point_b[0],
        point_a[1] - point_b[1],
        point_a[2] - point_b[2],
    )

    vector_bc = (
        point_c[0] - point_b[0],
        point_c[1] - point_b[1],
        point_c[2] - point_b[2],
    )

    magnitude_ba = math.sqrt(
        vector_ba[0] ** 2 +
        vector_ba[1] ** 2 +
        vector_ba[2] ** 2
    )

    magnitude_bc = math.sqrt(
        vector_bc[0] ** 2 +
        vector_bc[1] ** 2 +
        vector_bc[2] ** 2
    )

    if magnitude_ba == 0:
        return math.nan

    if magnitude_bc == 0:
        return math.nan

    dot_product = (
        vector_ba[0] * vector_bc[0] +
        vector_ba[1] * vector_bc[1] +
        vector_ba[2] * vector_bc[2]
    )

    cosine_angle = (
        dot_product /
        (magnitude_ba * magnitude_bc)
    )

    # Protect against floating point errors.
    cosine_angle = max(
        -1.0,
        min(1.0, cosine_angle)
    )

    angle_radians = math.acos(
        cosine_angle
    )

    angle_degrees = math.degrees(
        angle_radians
    )

    return angle_degrees


# ------------------------------------------------------------


def vector_difference(point_a, point_b):
    """
    Calculate:

        point_a - point_b
    """

    return (
        point_a[0] - point_b[0],
        point_a[1] - point_b[1],
        point_a[2] - point_b[2],
    )


# ------------------------------------------------------------


def vector_magnitude(vector):
    """
    Calculate magnitude of a 3D vector.
    """

    return math.sqrt(
        vector[0] ** 2 +
        vector[1] ** 2 +
        vector[2] ** 2
    )


# ------------------------------------------------------------


def calculate_velocity(
    current_position,
    previous_position,
    delta_time
):
    """
    Calculate 3D linear velocity magnitude.

        velocity_vector =
            (current_position - previous_position)
            / delta_time

        velocity =
            |velocity_vector|

    Returns:
        scalar velocity magnitude
    """

    if not is_valid_point(current_position):
        return math.nan

    if not is_valid_point(previous_position):
        return math.nan

    if delta_time <= 0:
        return math.nan

    displacement = vector_difference(
        current_position,
        previous_position
    )

    velocity_vector = (
        displacement[0] / delta_time,
        displacement[1] / delta_time,
        displacement[2] / delta_time,
    )

    return vector_magnitude(
        velocity_vector
    )


# ------------------------------------------------------------


def calculate_acceleration(
    current_velocity_vector,
    previous_velocity_vector,
    delta_time
):
    """
    Calculate 3D linear acceleration magnitude.

        acceleration_vector =
            (current_velocity_vector
             - previous_velocity_vector)
            / delta_time

        acceleration =
            |acceleration_vector|

    IMPORTANT:
        This is acceleration derived from the
        3D velocity vector, NOT from speed difference.
    """

    if current_velocity_vector is None:
        return math.nan

    if previous_velocity_vector is None:
        return math.nan

    if delta_time <= 0:
        return math.nan

    acceleration_vector = (

        (
            current_velocity_vector[0]
            - previous_velocity_vector[0]
        ) / delta_time,

        (
            current_velocity_vector[1]
            - previous_velocity_vector[1]
        ) / delta_time,

        (
            current_velocity_vector[2]
            - previous_velocity_vector[2]
        ) / delta_time,
    )

    return vector_magnitude(
        acceleration_vector
    )


# ------------------------------------------------------------


def calculate_velocity_vector(
    current_position,
    previous_position,
    delta_time
):
    """
    Calculate full 3D velocity vector.

    Returns:
        (vx, vy, vz)
    """

    if not is_valid_point(current_position):
        return None

    if not is_valid_point(previous_position):
        return None

    if delta_time <= 0:
        return None

    return (

        (
            current_position[0]
            - previous_position[0]
        ) / delta_time,

        (
            current_position[1]
            - previous_position[1]
        ) / delta_time,

        (
            current_position[2]
            - previous_position[2]
        ) / delta_time,
    )


# ============================================================
# EXTRACT 12 LANDMARKS
# ============================================================

def extract_landmarks(pose_world_landmarks):
    """
    Extract the 12 required 3D world landmarks.

    If a landmark is unavailable, its value is None.

    Returns:
        dictionary
    """

    landmarks = {}

    for landmark_id in LANDMARK_ORDER:

        landmark = (
            pose_world_landmarks.landmark[
                landmark_id
            ]
        )

        # MediaPipe visibility.
        if landmark.visibility < 0.1:

            landmarks[landmark_id] = None

        else:

            landmarks[landmark_id] = (
                landmark.x,
                landmark.y,
                landmark.z,
            )

    return landmarks


# ============================================================
# DISTANCE FEATURES
# ============================================================

def calculate_distance_features(
    landmarks
):
    """
    Calculate the 10 body distance features.
    """

    features = {}

    features["Shoulder_Width"] = distance(
        landmarks[11],
        landmarks[12]
    )

    features["Hip_Width"] = distance(
        landmarks[23],
        landmarks[24]
    )

    features["Left_Upper_Arm_Length"] = distance(
        landmarks[11],
        landmarks[13]
    )

    features["Right_Upper_Arm_Length"] = distance(
        landmarks[12],
        landmarks[14]
    )

    features["Left_Forearm_Length"] = distance(
        landmarks[13],
        landmarks[15]
    )

    features["Right_Forearm_Length"] = distance(
        landmarks[14],
        landmarks[16]
    )

    features["Left_Thigh_Length"] = distance(
        landmarks[23],
        landmarks[25]
    )

    features["Right_Thigh_Length"] = distance(
        landmarks[24],
        landmarks[26]
    )

    features["Left_Shin_Length"] = distance(
        landmarks[25],
        landmarks[27]
    )

    features["Right_Shin_Length"] = distance(
        landmarks[26],
        landmarks[28]
    )

    return features


# ============================================================
# ANGLE FEATURES
# ============================================================

def calculate_angle_features(
    landmarks
):
    """
    Calculate the 8 3D joint angles.
    """

    angles = {}

    # --------------------------------------------------------
    # ELBOWS
    # --------------------------------------------------------

    angles["Left Elbow"] = calculate_angle(
        landmarks[11],   # shoulder
        landmarks[13],   # elbow
        landmarks[15],   # wrist
    )

    angles["Right Elbow"] = calculate_angle(
        landmarks[12],
        landmarks[14],
        landmarks[16],
    )

    # --------------------------------------------------------
    # SHOULDERS
    # --------------------------------------------------------

    angles["Left Shoulder"] = calculate_angle(
        landmarks[13],   # elbow
        landmarks[11],   # shoulder
        landmarks[23],   # hip
    )

    angles["Right Shoulder"] = calculate_angle(
        landmarks[14],
        landmarks[12],
        landmarks[24],
    )

    # --------------------------------------------------------
    # HIPS
    # --------------------------------------------------------

    angles["Left Hip"] = calculate_angle(
        landmarks[11],   # shoulder
        landmarks[23],   # hip
        landmarks[25],   # knee
    )

    angles["Right Hip"] = calculate_angle(
        landmarks[12],
        landmarks[24],
        landmarks[26],
    )

    # --------------------------------------------------------
    # KNEES
    # --------------------------------------------------------

    angles["Left Knee"] = calculate_angle(
        landmarks[23],   # hip
        landmarks[25],   # knee
        landmarks[27],   # ankle
    )

    angles["Right Knee"] = calculate_angle(
        landmarks[24],
        landmarks[26],
        landmarks[28],
    )

    return angles


# ============================================================
# MAIN FEATURE EXTRACTION
# ============================================================

def extract_features_from_video():

    # ========================================================
    # CREATE OUTPUT DIRECTORIES
    # ========================================================

    csv_directory = os.path.dirname(
        OUTPUT_CSV
    )

    if csv_directory:
        os.makedirs(
            csv_directory,
            exist_ok=True
        )

    video_directory = os.path.dirname(
        OUTPUT_VIDEO
    )

    if video_directory:
        os.makedirs(
            video_directory,
            exist_ok=True
        )


    # ========================================================
    # CHECK INPUT VIDEO
    # ========================================================

    if not os.path.exists(VIDEO_SOURCE):

        raise FileNotFoundError(
            f"\nVideo not found:\n"
            f"{VIDEO_SOURCE}\n\n"
            f"Put your video inside:\n"
            f"data/raw/\n"
        )


    # ========================================================
    # OPEN VIDEO
    # ========================================================

    cap = cv2.VideoCapture(
        VIDEO_SOURCE
    )

    if not cap.isOpened():

        raise RuntimeError(
            "\nERROR: Video could not be opened.\n\n"
            "If you see:\n"
            "'cv2 has no attribute VideoCapture'\n"
            "then the problem is your OpenCV installation, "
            "not this feature calculation code.\n"
        )


    # ========================================================
    # VIDEO INFORMATION
    # ========================================================

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    frame_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    frame_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    default_dt = 1.0 / fps


    print()
    print("=" * 70)
    print("3D HUMAN POSE FEATURE EXTRACTION")
    print("=" * 70)

    print(
        f"Input video       : {VIDEO_SOURCE}"
    )

    print(
        f"FPS               : {fps:.3f}"
    )

    print(
        f"Frame count       : {total_frames}"
    )

    print(
        f"Resolution        : "
        f"{frame_width} x {frame_height}"
    )

    print(
        f"Time step         : {default_dt:.6f} sec"
    )

    print(
        f"Expected features : "
        f"{EXPECTED_FEATURE_COUNT}"
    )

    print("=" * 70)
    print()


    # ========================================================
    # VIDEO WRITER
    # ========================================================

    video_writer = None

    if SAVE_ANNOTATED_VIDEO:

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        video_writer = cv2.VideoWriter(
            OUTPUT_VIDEO,
            fourcc,
            fps,
            (
                frame_width,
                frame_height
            )
        )

        if not video_writer.isOpened():

            print(
                "WARNING: Could not create "
                "annotated output video."
            )

            video_writer = None


    # ========================================================
    # PREVIOUS LANDMARK INFORMATION
    # ========================================================

    previous_positions = {}

    previous_position_time = {}


    # ========================================================
    # PREVIOUS VELOCITY INFORMATION
    # ========================================================

    previous_velocity_vectors = {}

    previous_velocity_time = {}


    # ========================================================
    # PREVIOUS ANGLES
    # ========================================================

    previous_angles = {}

    previous_angle_time = {}


    # ========================================================
    # PREVIOUS ANGULAR VELOCITIES
    # ========================================================

    previous_angular_velocity = {}

    previous_angular_velocity_time = {}


    # ========================================================
    # CSV
    # ========================================================

    csv_file = open(
        OUTPUT_CSV,
        mode="w",
        newline="",
        encoding="utf-8"
    )

    writer = csv.writer(
        csv_file
    )

    header = [
        "Frame Id",
        "Timestamp",
    ] + feature_names

    writer.writerow(header)


    # ========================================================
    # COUNTERS
    # ========================================================

    frame_counter = 0

    detected_frames = 0

    written_rows = 0


    # ========================================================
    # MEDIAPIPE
    # ========================================================

    pose = mp_pose.Pose(

        static_image_mode=False,

        model_complexity=MODEL_COMPLEXITY,

        smooth_landmarks=True,

        enable_segmentation=False,

        min_detection_confidence=
        MIN_DETECTION_CONFIDENCE,

        min_tracking_confidence=
        MIN_TRACKING_CONFIDENCE
    )


    # ========================================================
    # MAIN LOOP
    # ========================================================

    try:

        while True:

            ret, frame = cap.read()

            if not ret:

                print(
                    "\nVideo finished."
                )

                break


            frame_counter += 1


            # ------------------------------------------------
            # Timestamp
            # ------------------------------------------------

            timestamp = (
                (frame_counter - 1)
                / fps
            )


            # ------------------------------------------------
            # BGR -> RGB
            # ------------------------------------------------

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            # ------------------------------------------------
            # MediaPipe
            # ------------------------------------------------

            results = pose.process(
                rgb_frame
            )


            # ------------------------------------------------
            # Initialize all features as NaN
            #
            # This is important:
            #
            # Every video frame gets one row.
            #
            # Missing pose does NOT delete the frame.
            # ------------------------------------------------

            features = [
                math.nan
                for _ in feature_names
            ]


            # =================================================
            # POSE DETECTED
            # =================================================

            if (
                results.pose_world_landmarks
                is not None
            ):

                detected_frames += 1


                # ------------------------------------------------
                # Extract landmarks
                # ------------------------------------------------

                landmarks = extract_landmarks(
                    results.pose_world_landmarks
                )


                # =================================================
                # 1. 3D COORDINATES
                # =================================================

                coordinate_values = []

                for landmark_id in LANDMARK_ORDER:

                    point = landmarks[
                        landmark_id
                    ]

                    if point is None:

                        coordinate_values.extend([
                            math.nan,
                            math.nan,
                            math.nan
                        ])

                    else:

                        coordinate_values.extend([
                            point[0],
                            point[1],
                            point[2]
                        ])


                # =================================================
                # 2. DISTANCES
                # =================================================

                distance_features = (
                    calculate_distance_features(
                        landmarks
                    )
                )


                distance_values = [

                    distance_features[
                        "Shoulder_Width"
                    ],

                    distance_features[
                        "Hip_Width"
                    ],

                    distance_features[
                        "Left_Upper_Arm_Length"
                    ],

                    distance_features[
                        "Right_Upper_Arm_Length"
                    ],

                    distance_features[
                        "Left_Forearm_Length"
                    ],

                    distance_features[
                        "Right_Forearm_Length"
                    ],

                    distance_features[
                        "Left_Thigh_Length"
                    ],

                    distance_features[
                        "Right_Thigh_Length"
                    ],

                    distance_features[
                        "Left_Shin_Length"
                    ],

                    distance_features[
                        "Right_Shin_Length"
                    ],
                ]


                # =================================================
                # 3. JOINT ANGLES
                # =================================================

                angle_features = (
                    calculate_angle_features(
                        landmarks
                    )
                )


                angle_values = [

                    angle_features[
                        "Left Elbow"
                    ],

                    angle_features[
                        "Right Elbow"
                    ],

                    angle_features[
                        "Left Shoulder"
                    ],

                    angle_features[
                        "Right Shoulder"
                    ],

                    angle_features[
                        "Left Hip"
                    ],

                    angle_features[
                        "Right Hip"
                    ],

                    angle_features[
                        "Left Knee"
                    ],

                    angle_features[
                        "Right Knee"
                    ],
                ]


                # =================================================
                # 4. LINEAR VELOCITY
                # =================================================

                velocity_values = []

                current_velocity_vectors = {}


                for landmark_id in LANDMARK_ORDER:

                    current_position = (
                        landmarks[landmark_id]
                    )


                    if (
                        current_position is not None
                        and landmark_id
                        in previous_positions
                    ):

                        previous_position = (
                            previous_positions[
                                landmark_id
                            ]
                        )

                        previous_time = (
                            previous_position_time[
                                landmark_id
                            ]
                        )

                        delta_time = (
                            timestamp
                            - previous_time
                        )


                        velocity_vector = (
                            calculate_velocity_vector(
                                current_position,
                                previous_position,
                                delta_time
                            )
                        )


                        if velocity_vector is not None:

                            current_velocity_vectors[
                                landmark_id
                            ] = velocity_vector

                            velocity_magnitude = (
                                vector_magnitude(
                                    velocity_vector
                                )
                            )

                        else:

                            velocity_magnitude = (
                                math.nan
                            )

                    else:

                        velocity_magnitude = (
                            math.nan
                        )


                    velocity_values.append(
                        velocity_magnitude
                    )


                # =================================================
                # 5. LINEAR ACCELERATION
                # =================================================

                acceleration_values = []


                for landmark_id in LANDMARK_ORDER:

                    acceleration = math.nan


                    if (
                        landmark_id
                        in current_velocity_vectors
                        and landmark_id
                        in previous_velocity_vectors
                    ):

                        current_velocity = (
                            current_velocity_vectors[
                                landmark_id
                            ]
                        )

                        previous_velocity = (
                            previous_velocity_vectors[
                                landmark_id
                            ]
                        )

                        previous_time = (
                            previous_velocity_time[
                                landmark_id
                            ]
                        )

                        delta_time = (
                            timestamp
                            - previous_time
                        )


                        acceleration = (
                            calculate_acceleration(
                                current_velocity,
                                previous_velocity,
                                delta_time
                            )
                        )


                    acceleration_values.append(
                        acceleration
                    )


                # =================================================
                # 6. ANGULAR VELOCITY
                # =================================================

                angular_velocity_values = []

                current_angular_velocity = {}


                for joint in ANGLE_ORDER:

                    angle = angle_features[
                        joint
                    ]

                    angular_velocity = (
                        math.nan
                    )


                    if (
                        math.isfinite(angle)
                        and joint
                        in previous_angles
                    ):

                        previous_angle = (
                            previous_angles[
                                joint
                            ]
                        )

                        previous_time = (
                            previous_angle_time[
                                joint
                            ]
                        )

                        delta_time = (
                            timestamp
                            - previous_time
                        )


                        if (
                            math.isfinite(
                                previous_angle
                            )
                            and delta_time > 0
                        ):

                            angular_velocity = (
                                (
                                    angle
                                    - previous_angle
                                )
                                / delta_time
                            )


                            current_angular_velocity[
                                joint
                            ] = angular_velocity


                    angular_velocity_values.append(
                        angular_velocity
                    )


                # =================================================
                # 7. ANGULAR ACCELERATION
                # =================================================

                angular_acceleration_values = []


                for joint in ANGLE_ORDER:

                    angular_acceleration = (
                        math.nan
                    )


                    if (
                        joint
                        in current_angular_velocity
                        and joint
                        in previous_angular_velocity
                    ):

                        current_omega = (
                            current_angular_velocity[
                                joint
                            ]
                        )

                        previous_omega = (
                            previous_angular_velocity[
                                joint
                            ]
                        )

                        previous_time = (
                            previous_angular_velocity_time[
                                joint
                            ]
                        )

                        delta_time = (
                            timestamp
                            - previous_time
                        )


                        if delta_time > 0:

                            angular_acceleration = (
                                (
                                    current_omega
                                    - previous_omega
                                )
                                / delta_time
                            )


                    angular_acceleration_values.append(
                        angular_acceleration
                    )


                # =================================================
                # COMBINE ALL 94 FEATURES
                # =================================================

                features = (

                    coordinate_values

                    + distance_values

                    + angle_values

                    + velocity_values

                    + acceleration_values

                    + angular_velocity_values

                    + angular_acceleration_values
                )


                # =================================================
                # CHECK FEATURE COUNT
                # =================================================

                if len(features) != EXPECTED_FEATURE_COUNT:

                    raise RuntimeError(
                        "\nFeature count mismatch!\n"
                        f"Expected: "
                        f"{EXPECTED_FEATURE_COUNT}\n"
                        f"Got: {len(features)}"
                    )


                # =================================================
                # UPDATE PREVIOUS LANDMARK DATA
                # =================================================

                for landmark_id in LANDMARK_ORDER:

                    current_position = (
                        landmarks[
                            landmark_id
                        ]
                    )


                    if current_position is not None:

                        previous_positions[
                            landmark_id
                        ] = current_position

                        previous_position_time[
                            landmark_id
                        ] = timestamp


                # =================================================
                # UPDATE PREVIOUS VELOCITY DATA
                # =================================================

                for landmark_id, velocity_vector in (
                    current_velocity_vectors.items()
                ):

                    previous_velocity_vectors[
                        landmark_id
                    ] = velocity_vector

                    previous_velocity_time[
                        landmark_id
                    ] = timestamp


                # =================================================
                # UPDATE PREVIOUS ANGLE DATA
                # =================================================

                for joint in ANGLE_ORDER:

                    angle = angle_features[
                        joint
                    ]


                    if math.isfinite(angle):

                        previous_angles[
                            joint
                        ] = angle

                        previous_angle_time[
                            joint
                        ] = timestamp


                # =================================================
                # UPDATE PREVIOUS ANGULAR VELOCITY
                # =================================================

                for joint, omega in (
                    current_angular_velocity.items()
                ):

                    previous_angular_velocity[
                        joint
                    ] = omega

                    previous_angular_velocity_time[
                        joint
                    ] = timestamp


            # =================================================
            # WRITE CSV ROW
            # =================================================

            row = [

                frame_counter,

                timestamp,

            ] + features


            writer.writerow(
                row
            )

            written_rows += 1


            # =================================================
            # DRAW SKELETON
            # =================================================

            if (
                results.pose_landmarks
                is not None
            ):

                mp_drawing.draw_landmarks(

                    frame,

                    results.pose_landmarks,

                    mp_pose.POSE_CONNECTIONS
                )


            # =================================================
            # DISPLAY INFORMATION
            # =================================================

            status_text = (
                "POSE DETECTED"
                if results.pose_world_landmarks
                is not None
                else
                "NO POSE"
            )


            status_color = (
                (0, 255, 0)
                if results.pose_world_landmarks
                is not None
                else
                (0, 0, 255)
            )


            cv2.putText(

                frame,

                f"Frame: {frame_counter}",

                (20, 35),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255, 255, 255),

                2
            )


            cv2.putText(

                frame,

                f"Time: {timestamp:.3f}s",

                (20, 65),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255, 255, 255),

                2
            )


            cv2.putText(

                frame,

                status_text,

                (20, 95),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                status_color,

                2
            )


            # =================================================
            # SAVE ANNOTATED VIDEO
            # =================================================

            if video_writer is not None:

                video_writer.write(
                    frame
                )


            # =================================================
            # SHOW VIDEO
            # =================================================

            if SHOW_VIDEO:

                cv2.imshow(
                    "3D Human Pose - Feature Extraction",
                    frame
                )


                key = cv2.waitKey(1) & 0xFF


                # Q = quit
                if key == ord("q"):

                    print(
                        "\nProcessing stopped "
                        "by user."
                    )

                    break


            # =================================================
            # PROGRESS
            # =================================================

            if (
                frame_counter % 30 == 0
                or frame_counter == total_frames
            ):

                if total_frames > 0:

                    progress = (
                        frame_counter
                        / total_frames
                    ) * 100

                    print(
                        f"Processing: "
                        f"{frame_counter}/"
                        f"{total_frames} "
                        f"({progress:.1f}%)"
                    )

                else:

                    print(
                        f"Processed frames: "
                        f"{frame_counter}"
                    )


    finally:

        # ====================================================
        # CLEANUP
        # ====================================================

        pose.close()

        cap.release()

        if video_writer is not None:

            video_writer.release()

        csv_file.close()

        cv2.destroyAllWindows()


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    detection_percentage = 0.0


    if frame_counter > 0:

        detection_percentage = (
            detected_frames
            / frame_counter
        ) * 100


    print()
    print("=" * 70)
    print("FEATURE EXTRACTION COMPLETED")
    print("=" * 70)

    print(
        f"Frames processed      : "
        f"{frame_counter}"
    )

    print(
        f"CSV rows written      : "
        f"{written_rows}"
    )

    print(
        f"Pose detected frames  : "
        f"{detected_frames}"
    )

    print(
        f"Detection percentage  : "
        f"{detection_percentage:.2f}%"
    )

    print(
        f"Features per frame    : "
        f"{EXPECTED_FEATURE_COUNT}"
    )

    print(
        f"CSV output            : "
        f"{OUTPUT_CSV}"
    )

    if SAVE_ANNOTATED_VIDEO:

        print(
            f"Video output          : "
            f"{OUTPUT_VIDEO}"
        )

    print("=" * 70)
    print()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    extract_features_from_video()
"""Feature contracts shared by extraction, matrix building, and training."""

AFFECT_FEATURE_SCHEMA = "retinaface_bytetrack_fer_v1"
INTERACTION_FEATURE_SCHEMA = "yolov8_pose_sampledtrack_collective32_v3"
MULTI_BRANCH_FEATURE_SCHEMA = "scene576_interaction32_sampledtrack_affect8_v3"
BEHAVIORAL_FEATURE_SCHEMA = "interaction32_sampledtrack_affect8_v3"

INTERACTION_COLUMNS = (
    "raised_head_ratio", "lowered_head_ratio", "head_turned_ratio", "mean_orientation_to_target",
    "orientation_consensus_ratio", "std_orientation_score", "min_orientation_score", "max_orientation_score",
    "upright_ratio", "forward_lean_proxy_ratio", "very_low_head_proxy_ratio", "mean_posture_elevation",
    "min_posture_elevation", "std_posture_elevation", "shoulder_tilt_mean", "posture_symmetry",
    "student_count_norm", "mean_face_visibility", "hidden_face_ratio", "both_eyes_visible_ratio",
    "single_eye_ratio", "median_face_visibility", "face_visibility_iqr", "posture_observation_ratio",
    "matched_center_displacement_mean_x10", "matched_center_displacement_std_x10",
    "matched_stillness_ratio", "matched_high_motion_ratio", "speaker_in_zone",
    "matched_speaker_displacement_x10", "orientation_alignment", "head_elevation_dispersion",
)

AFFECT_COLUMNS = (
    "anger",
    "disgust",
    "fear",
    "happiness",
    "sadness",
    "surprise",
    "neutral",
    "affect_reliability",
)

MULTI_BRANCH_SHAPE = (8, 616)
BEHAVIORAL_SHAPE = (8, 40)

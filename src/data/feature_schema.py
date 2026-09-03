"""Feature contracts shared by extraction, matrix building, and training."""

AFFECT_FEATURE_SCHEMA = "retinaface_bytetrack_fer_v1"
INTERACTION_FEATURE_SCHEMA = "yolov8_person_geometry_32_v1"
TRACK_INTERACTION_FEATURE_SCHEMA = "yolov8_pose_bytetrack_role_pool_40_v1"
MULTI_BRANCH_FEATURE_SCHEMA = "scene576_interaction32_affect8_track_v1"
LEGACY_BEHAVIORAL_FEATURE_SCHEMA = "interaction32_affect8_track_v1"
BEHAVIORAL_FEATURE_SCHEMA = "interaction_track40_affect8_track_v1"

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

TRACK_INTERACTION_COLUMNS = (
    "teacher_present",
    "teacher_position_x",
    "teacher_position_y",
    "teacher_track_coverage",
    "teacher_detection_confidence",
    "teacher_motion",
    "teacher_role_confidence",
    "teacher_frame_reliability",
    "visible_student_count",
    "valid_student_track_count",
    "mean_student_x",
    "mean_student_y",
    "student_spread_x",
    "student_spread_y",
    "mean_student_width",
    "mean_student_height",
    "mean_student_track_coverage",
    "mean_student_detection_confidence",
    "mean_student_pose_confidence",
    "mean_student_motion",
    "instruction_alignment_proxy",
    "instruction_alignment_reliability",
    "instruction_aligned_fraction",
    "instruction_aligned_fraction_reliability",
    "student_interaction_reliability",
    "mean_pairwise_student_distance",
    "std_pairwise_student_distance",
    "min_pairwise_student_distance",
    "close_student_pair_fraction",
    "student_cluster_compactness",
    "pairwise_relation_reliability",
    "mean_student_teacher_distance",
    "student_teacher_distance_reliability",
    "peer_aligned_fraction",
    "peer_alignment_reliability",
    "visible_unknown_track_count",
    "visible_untracked_detection_count",
    "total_clip_track_count",
    "student_visibility_fraction",
    "frame_detection_reliability",
)

MULTI_BRANCH_SHAPE = (8, 616)
TRACK_INTERACTION_SHAPE = (8, len(TRACK_INTERACTION_COLUMNS))
BEHAVIORAL_SHAPE = (8, len(TRACK_INTERACTION_COLUMNS) + len(AFFECT_COLUMNS))

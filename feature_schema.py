"""Feature contracts shared by extraction, matrix building, and training."""

AFFECT_FEATURE_SCHEMA = "retinaface_bytetrack_fer_v1"
INTERACTION_FEATURE_SCHEMA = "yolov8_person_geometry_32_v1"
MULTI_BRANCH_FEATURE_SCHEMA = "scene576_interaction32_affect8_track_v1"
BEHAVIORAL_FEATURE_SCHEMA = "interaction32_affect8_track_v1"

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

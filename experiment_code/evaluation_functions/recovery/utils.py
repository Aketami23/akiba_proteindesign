from config_utils import load_config

def calculate_recovery(query_sequence: str, config_path: str) -> float:
    _config = load_config(config_path)
    known_sequence = _config["known_sequence"]["sequence"]
    # proteinMPNN_prediction_seq = config["known_sequence"]["mpnn_predict_sequence"]
    known_seq = known_sequence
    seq_length = len(known_seq)
    matches = sum(s1 == s2 for s1, s2 in zip(query_sequence, known_seq))
    recovery_rate = matches / seq_length if seq_length > 0 else 0
    return recovery_rate
import random
from protein_mpnn.utils import initialize_model, run_inference

def mutation(sequence: str, number: int, pop_count: int) -> tuple[str, str, None]:
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    results = []
    new_header = f"1QYS-Chain_A-TOP7-round_{number+ pop_count}"
    print(f"new_header: {new_header}")

    seq_list = list(sequence)
    pos = random.randint(0, len(seq_list) - 1)
    seq_list[pos] = random.choice([aa for aa in amino_acids if aa != seq_list[pos]])
    mutated_sequence = "".join(seq_list)
    results = (new_header, mutated_sequence, None)
    
    return results

def mutation_with_mpnn(config_path: str, sequence: str, number: int, pop_count: int, model:str, device: any) -> tuple[str, str, None]:
    from config_utils import load_config
    _config = load_config(config_path)
    result = ()
    new_header = f"1QYS-Chain_A-TOP7-round_{number + pop_count}"
    num_seq_per_target = 1
    sampling_temp = _config["experiment_config"]["mpnn_temperature"]
    seq_length = _config["known_sequence"]["length"]
    batch_size = 1
    seq_length = len(sequence)
    random.seed(None)
    seed = random.randint(0, 2**32 - 1)
    random_position = random.randint(1, seq_length)
    design_only_positions = []
    for pos in [random_position - 1, random_position, random_position + 1]:
        if 1 <= pos <= seq_length:
            design_only_positions.append(pos)

    mutated_sequence = run_inference(config_path, model, sequence, design_only_positions, num_seq_per_target, sampling_temp, seed, batch_size, device)
    result = (new_header, mutated_sequence, None)
    return result

def generate_offspring(solution: any, count: int)-> list[tuple[str, str, None]]:
    new_queries = []
    pop_count = 1
    for i in solution:
        new_queries.append(mutation(i, count, pop_count))
        pop_count += 1
    return new_queries

def generate_offspring_npmm(solution: any, count: int, config_path: str) -> list[tuple[str, str, None]]:
    new_queries = []
    pop_count = 1
    random.seed(None)
    seed = random.randint(0, 2**32 - 1)
    model, device = initialize_model(config_path, seed)
    for i in solution:
        new_queries.append(mutation_with_mpnn(config_path, i, count, pop_count, model, device))
        pop_count += 1
    return new_queries

def generate_random_sequence_list(seq_length: int, num_sequences: int) -> list[tuple[str, str, None]]:
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    result = []
    for i in range(1, num_sequences + 1):
        ## ハードコーディング
        seq_id = f"1QYS-Chain_A-TOP7-round_{i}"
        sequence = ''.join(random.choices(amino_acids, k=seq_length))
        result.append((seq_id, sequence, None))
    return result

    # return example
    """
    return example

    queries = [
    ('1QYS-Chain_A-TOP7-round_1', 'YHINMYCCDFKRSRVHHFTFMAGDWWFDSGCKKMLWNMCYKPVQDHIHVMVSYYQWYKHRFVLCITHYIFINWYMPTWMSLDSYAVTYRITM', None), 
    ('1QYS-Chain_A-TOP7-round_2', 'MLQCIIHMKCGSRTYIKYAQAEFQELVWNKKACNLHLPQHSWMHFLMWPSEEAPWGEIPAKFHVLRAWVKEICPWCYYVDDNSIDCMWIRTI', None)
    ]
    """
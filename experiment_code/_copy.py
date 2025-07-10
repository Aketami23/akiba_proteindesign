import torch
import random
import numpy as np
from protein_mpnn.protein_mpnn_utils import ProteinMPNN
from protein_mpnn.protein_mpnn_utils import _scores, _S_to_seq, tied_featurize
from config_utils import load_config
import copy

def initialize_model(seed=37, device=None):
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)   
    
    hidden_dim = 128
    num_layers = 3 
    checkpoint_path = "/Users/ryo/projects/akiba_project/proteinMPNN/test/vanilla_model_weight/v_48_020.pt"
    
    if device is None:
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device) 
    backbone_noise = 0.00
    model = ProteinMPNN(ca_only=False, num_letters=21, node_features=hidden_dim, 
                        edge_features=hidden_dim, hidden_dim=hidden_dim, 
                        num_encoder_layers=num_layers, num_decoder_layers=num_layers, 
                        augment_eps=backbone_noise, k_neighbors=checkpoint['num_edges'])
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device

def run_inference(config_path: str, model:any, init_seq: str, design_only_positions: list[int], num_seq_per_target: int, sampling_temp: float, seed: int, batch_size: int = 1, device: any = None)-> str:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)   

    _config = load_config(config_path)
    
    if device is None:
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    
    NUM_BATCHES = num_seq_per_target//batch_size
    BATCH_COPIES = batch_size
    temperatures = [float(sampling_temp)]
    omit_AAs_list = []
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'   
    omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)
    chain_id_dict = {'1qys': [['A'], []]}
    fix_positions = [i for i in range(1, _config['known_sequence']["length"]+1) if i not in design_only_positions]
    fixed_positions_dict = {'1qys': {'A': fix_positions}}
    
    pssm_dict = None
    omit_AA_dict = None
    tied_positions_dict = None
    bias_by_res_dict = None
    bias_AAs_np = np.zeros(len(alphabet))
    # wild_type = "DIQVQVNIDDNGKNFDYTYTVTTESELQKVLNELMDYIKKQGAKRVRISITARTKKEAEKFAAILIKVFAELGYNDINVTFDGDTVTVEGQL"
    known_seq = _config["known_sequence"]["sequence"]
    protein = {'seq_chain_A': init_seq, 
               'coords_chain_A': _config["target_structure"]["formated_coordinate"], 
               'name': '1qys', 
               'num_of_chains': 1, 
               'seq': init_seq}

    # Validation epoch
    with torch.no_grad():
        all_probs_list = []
        all_log_probs_list = []
        S_sample_list = []
        batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
        X, S, mask, _, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(batch_clones, device, chain_id_dict, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict, ca_only=False)
        pssm_log_odds_mask = (pssm_log_odds_all > 0.0).float() #1.0 for true, 0.0 for false
        randn_1 = torch.randn(chain_M.shape, device=X.device)
        log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
        mask_for_loss = mask*chain_M*chain_M_pos
        scores = _scores(S, log_probs, mask_for_loss) #score only the redesigned part
        global_scores = _scores(S, log_probs, mask) #score the whole structure-sequence
        
        # Generate sequences
        for temp in temperatures:
            for j in range(NUM_BATCHES):
                randn_2 = torch.randn(chain_M.shape, device=X.device)
                sample_dict = model.sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=0.0, pssm_log_odds_flag=bool(0), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(0), bias_by_res=bias_by_res_all)
                S_sample = sample_dict["S"] 
                    
                log_probs = model(X, S_sample, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_2, use_input_decoding_order=True, decoding_order=sample_dict["decoding_order"])
                mask_for_loss = mask*chain_M*chain_M_pos
                scores = _scores(S_sample, log_probs, mask_for_loss)
                scores = scores.cpu().data.numpy()
                
                global_scores = _scores(S_sample, log_probs, mask) #score the whole structure-sequence
                global_scores = global_scores.cpu().data.numpy()
                
                all_probs_list.append(sample_dict["probs"].cpu().data.numpy())
                all_log_probs_list.append(log_probs.cpu().data.numpy())
                S_sample_list.append(S_sample.cpu().data.numpy())
                for b_ix in range(BATCH_COPIES):
                    seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
    return seq


def calculate_sequence_diversity(sequences):
    """
    配列の多様性を計算する関数
    
    Args:
        sequences (list): 生成された配列のリスト
    
    Returns:
        dict: 多様性の指標（平均ハミング距離、位置ごとのエントロピーなど）
    """
    import numpy as np
    from collections import Counter
    import math

    # 配列が少なくとも2つないと多様性を計算できない
    if len(sequences) < 2:
        return {"avg_hamming_distance": 0, "position_entropy": [], "mean_entropy": 0}
    
    # 平均ハミング距離を計算
    total_distance = 0
    pair_count = 0
    
    for i in range(len(sequences)):
        for j in range(i+1, len(sequences)):
            hamming_dist = sum(s1 != s2 for s1, s2 in zip(sequences[i], sequences[j]))
            total_distance += hamming_dist
            pair_count += 1
    
    avg_hamming_distance = total_distance / pair_count if pair_count > 0 else 0
    
    # 位置ごとのエントロピーを計算
    seq_length = len(sequences[0])
    position_entropy = []
    
    for pos in range(seq_length):
        # 各位置でのアミノ酸の頻度を数える
        position_aas = [seq[pos] for seq in sequences]
        aa_counts = Counter(position_aas)
        
        # エントロピーを計算
        entropy = 0
        for aa, count in aa_counts.items():
            prob = count / len(sequences)
            entropy -= prob * math.log2(prob)
        
        position_entropy.append(entropy)
    
    mean_entropy = sum(position_entropy) / len(position_entropy)
    
    return {
        "avg_hamming_distance": avg_hamming_distance,
        "position_entropy": position_entropy,
        "mean_entropy": mean_entropy
    }


def main(jsonl_path, chain_id_jsonl, fixed_positions_jsonl, out_folder, num_seq_per_target, sampling_temp, seed, batch_size):
    model, device = initialize_model(seed)
    run_inference(model, jsonl_path, chain_id_jsonl, fixed_positions_jsonl, out_folder, 
                 num_seq_per_target, sampling_temp, seed, batch_size, device)

import random
import csv

if __name__ == "__main__":
    num_seq_per_target = 1
    sampling_temp = 0.3
    random.seed(None)
    seed = random.randint(0, 2**32 - 1)
    batch_size = 1
    init_seq = "GKWALFVFEFEDEVIVNNLVFHFPLCYHMPLVLELLRHHVREHPVASFSVKAKIGPRAQQEAILLRLKSMWLGYKRNTVTFMDPTVTLISVR"
    model, device = initialize_model(seed)
    config_path = "./experiment_code/config.yaml"
    for i in range(100):
        random.seed(None)
        seq_length = len(init_seq)
        random_position = random.randint(1, seq_length)
        design_only_positions = []
        for pos in [random_position - 1, random_position, random_position + 1]:
            if 1 <= pos <= seq_length:
                design_only_positions.append(pos)
        print(f"Design only position: {design_only_positions}") 

        seq = run_inference(config_path, model, init_seq, design_only_positions, num_seq_per_target, sampling_temp, seed, batch_size, device)
        init_seq = seq
        print(f"seq: {seq}")
    
    # recovery_rate = seq/_config["known_sequence"]["sequence"]
    _config = load_config(config_path)
    recovery_rate = sum(s1 == s2 for s1, s2 in zip(seq, _config["known_sequence"]["sequence"])) / len(init_seq) if len(init_seq) > 0 else 0
    print(f"Recovery rate: {recovery_rate:.2f}")
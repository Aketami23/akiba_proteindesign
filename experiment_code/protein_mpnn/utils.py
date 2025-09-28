import torch
import random
import numpy as np
import copy
from .protein_mpnn_utils import ProteinMPNN
from .protein_mpnn_utils import _scores, _S_to_seq, tied_featurize
from config_utils import load_config

def initialize_model(config_path: str, seed: int, device: any = None)-> tuple[any, any]:
    _config = load_config(config_path)
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)   
    
    hidden_dim = 128
    num_layers = 3 
    checkpoint_path = _config["protein_mpnn"]["checkpoint_path"]
    
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

def run_inference(config_path: str, 
                  model:any, init_seq: str, 
                  design_only_positions: list[int], 
                  num_seq_per_target: int, 
                  sampling_temp: float, 
                  seed: int, 
                  batch_size: int = 1, 
                  device: any = None)-> str:
    
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
    # known_seq = _config["known_sequence"]["sequence"]
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
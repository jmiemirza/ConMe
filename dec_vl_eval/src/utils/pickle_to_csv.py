import numpy as np
import pandas as pd

if __name__ == '__main__':
    # pickle_path = '/dccstor/leonidka1/irenespace/llm_results_updated/eval/llava-mix/ed_MODEL_SPEC_UPDATED_LLM_fd_llava-mix_ev_llava-v1.5-7b_wn_orig_neg_s_loss_score_t_0_nt_128_p_11/eval_model_spec/llava-mix_llava-v1.5-7b/'
    # pickle_path = '/dccstor/leonidka1/irenespace/aro_results/corrected_templates/m_instructblip_vicuna_7b_ed_ARO_VGA_s_loss_score/baseline/vga_instructblip_vicuna_7b-prompt-3'
    # pickle_path = '/dccstor/leonidka1/irenespace/sugar_results/corrected_templates/m_idefics2-8b_ed_SUGAR_s_generate/baseline/sugar_idefics2-8b/'
    pickle_path = '/dccstor/leonidka1/irenespace/gpt4v_results/desc/step7/ln_finetune/instructblip_flan_t5/generate/'
    df = pd.read_pickle(f'{pickle_path}/results_dataframe.pkl')
    df.to_csv(f'{pickle_path}/results_dataframe.csv')

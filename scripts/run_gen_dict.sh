saet -e


# # build dico
echo "-----------------------------Building dictionary..." 
# python scripts/cli.py generate_dico --sequencefile $1 --dictconf $2 --dictconflight $3 --reco 5.0 --echospacing 1.11 --TI 8.32 --isbuildphi True
python scripts/cli.py gen_dict_mrf --datafile $1 --force True --dest mrf_dict_optim --wait-time 2.99 --seqfile $2 --dictconf $3 --dictconf-light $4 --is_build_phi True

# # seq="/home/ahelleboid/Low_field_MRF/mrftools/dico/mrf_sequence_TE_min_FA_optim.json"
# dico="/home/ahelleboid/Low_field_MRF/mrftools/dico/mrf_dictconf_Dico2_Invivo_overshoot_0_55T_fT1.json"
# dico_light="/home/ahelleboid/Low_field_MRF/mrftools/dico/mrf_dictconf_Dico2_Invivo_light_for_matching_overshoot_0_55T_fT1.json"
# # python scripts/cli.py generate_dico --sequencefile $seq --dictconf $dico --dictconflight $dico_light --reco 5.0 --echospacing 1.89 --TI 8.32 --isbuildphi True
# # mv mrf_dict/*.pkl dico/dico_mrf_TE_min_v1_FA_optim.pkl

# # exit

# seq="/home/ahelleboid/Low_field_MRF/mrftools/dico/mrf_sequence_adjusted_0_55T_TEminv2_FA.json"
# python scripts/cli.py generate_dico --sequencefile $seq --dictconf $dico --dictconflight $dico_light --reco 5.0 --echospacing 1.89 --TI 8.32 --isbuildphi True
# mv mrf_dict/*.pkl dico/dico_mrf_TE_min_v2_FA.pkl

# seq="/home/ahelleboid/Low_field_MRF/mrftools/dico/mrf_sequence_TE_v2_FA_optim.json"
# python scripts/cli.py generate_dico --sequencefile $1 --dictconf $2 --dictconflight $3 --reco 5.0 --echospacing 1.89 --TI 8.32 --isbuildphi True
# mv mrf_dict/*.pkl dico/dico_mrf_TE_v2_FA_optim.pkl

# seq="/home/ahelleboid/Low_field_MRF/mrftools/dico/mrf_sequence_TE_v1_FA_optim.json"
# python scripts/cli.py generate_dico --sequencefile $1 --dictconf $2 --dictconflight $3 --reco 5.0 --echospacing 1.89 --TI 8.32 --isbuildphi True
# mv mrf_dict/*.pkl dico/dico_mrf_TE_v1_FA_optim.pkl

# seq="/home/ahelleboid/Low_field_MRF/mrftools/dico/mrf_sequence_adjusted_0_55T_TEv1.json"
# python scripts/cli.py generate_dico --sequencefile $1 --dictconf $2 --dictconflight $3 --reco 5.0 --echospacing 1.89 --TI 8.32 --isbuildphi True
# mv mrf_dict/*.pkl dico/dico_mrf_TE_v1_FA.pkl

# seq="/home/ahelleboid/Low_field_MRF/mrftools/dico/mrf_sequence_adjusted_0_55T_TEv1_1_4.json"
# python scripts/cli.py generate_dico --sequencefile $1 --dictconf $2 --dictconflight $3 --reco 5.0 --echospacing 1.89 --TI 8.32 --isbuildphi True
# mv mrf_dict/*.pkl dico/dico_mrf_TE_v1_FA_1_4.pkl
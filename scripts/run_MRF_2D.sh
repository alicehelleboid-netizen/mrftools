set -e

# # # # Extracting k-space 
echo "-----------------------------Extracting k-space data..."
python scripts/cli.py extract_data --filename  $1.dat

# # # Coil sensitivity maps estimation
echo "-----------------------------Estimating coil sensitivity maps..."
python scripts/cli.py calculate_sensi --filekdata data/3T/kdata.npy

# # volume reconstruction
echo "-----------------------------Reconstructing volumes..."
# python scripts/cli.py build_volumes --filekdata data/kdata.npy --fileb1 data/b1.npy
python scripts/cli.py build_volumes_singular --filekdata data/3T/kdata.npy --fileb1 data/3T/b1.npy --dictdir . --dictfiles $2 --L0 6 --niter 1 --regularizer wavelet --lambd 1e-4

# # # # Mask
# # echo "-----------------------------Building masks..."
# python scripts/cli.py build_masks_singular --filevolumes data/volumes_singular.npy --l 0 --threshold 0.04
# # python scripts/cli.py build_masks --filekdata data/kdata.npy --fileb1 data/b1.npy --threshold-factor 0.06 0.007

# # build maps
# echo "-----------------------------Building parameter maps..."
# python scripts/cli.py build_maps --filevolumes data/volumes_singular.npy --filemasks data/masks_singular.npy --fileseq data/dico_seqParams.pkl --dictfile $2 --volumestype singular --returncost True
# # python scripts/cli.py build_maps --filevolumes data/volumes.npy --filemasks data/masks.npy --fileseq data/dico_seqParams.pkl --dictfiles $2 --volumestype raw --returncost True

# # spec=""
# # base=$(basename "$f")
# # for f in data/*.mha; do
# #   base=$(basename "$f" .mha)
# #   cp "$f" "/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#5_2025_LF_MRF/3_Data_Processed/low_field/${1#data/}_${base}${spec}.mha"
# # done



#  plusieurs dico/image

# DICOS=('DICO_ref_055T_TEv1_FA5' 'dico_mrf_TE_v1_FA_optim' 'dico_mrf_TE_v2_FA_optim' 'dico_mrf_TE_min_v2_FA' 'dico_mrf_TE_minv2_FA_optim')
# DATA_FILES=( 
# "/home/ahelleboid/Low_field_MRF/mrftools/data/03022026/meas_MID00036_FID08575_raFin_FreeMax_TE_v1"
# "/home/ahelleboid/Low_field_MRF/mrftools/data/03022026/meas_MID00038_FID08577_raFin_FreeMax_TE_v1_FAoptim"
#  "/home/ahelleboid/Low_field_MRF/mrftools/data/03022026/meas_MID00044_FID08583_raFin_FreeMax_TE_v2_FAoptim" 
# "/home/ahelleboid/Low_field_MRF/mrftools/data/03022026/meas_MID00048_FID08587_raFin_FreeMax_TEmin_v2_FA" 
#  "/home/ahelleboid/Low_field_MRF/mrftools/data/03022026/meas_MID00049_FID08588_raFin_FreeMax_TEmin_v2_FA_optim")


# for i in "${!DICOS[@]}"

# do
#     DICO_FILE="${DICOS[$i]}"
#     DATA_FILE="${DATA_FILES[$i]}"

#     echo "Dico : $DICO_FILE"
#     echo "Data : $DATA_FILE"

#     echo "=========================================="
#     echo "Extracting k-space data..."
#     python scripts/cli.py extract_data --filename "$DATA_FILE.dat"

#     echo "=========================================="
#     echo "Estimating coil sensitivity maps..."
#     python scripts/cli.py calculate_sensi --filekdata data/03022026/kdata.npy

#     # Get dictionary name without extension for output folder
#     dico_name=$(basename "$DICO_FILE" .pkl)
    
#     # Create output directory for this dictionary
#     output_dir="results_$dico_name"
#     # mkdir -p "$output_dir"
    
#     # # Reconstruct singular volumes for this dictionary
#     echo "Reconstructing singular volumes for $dico_name..."
#     python scripts/cli.py build_volumes --filekdata data/03022026/kdata.npy --fileb1 data/03022026/b1.npy
    
#     # Copy and rename singular volumes
#     # cp data/03022026/volumes.npy "$output_dir/volumes_raw.npy"
    
#     # Build masks for singular volumes
#     echo "Building masks for singular volumes..."
#     python scripts/cli.py build_masks --filekdata data/03022026/kdata.npy --fileb1 data/03022026/b1.npy --threshold 0.06
    
#     # Copy and rename singular masks
#     # cp data/03022026/masks.npy "$output_dir/masks_raw.npy"
    
#     # # Build maps for this dictionary
#     echo "Building parameter maps for $dico_name..."
#     python scripts/cli.py build_maps --filevolumes "$output_dir/volumes_raw.npy" --filemasks "$output_dir/masks_raw.npy" --fileseq data/03022026/dico_seqParams.pkl --dictfile "dico/$DICO_FILE.pkl" --volumestype raw --returncost True
        
#     # Copy all results to output directory
#     cp data/03022026/maps.pkl "$output_dir/all_maps_singular_v2.pkl.npy"
#     cp /home/ahelleboid/Low_field_MRF/mrftools/matched_signals_coarse_dico.npy "$output_dir/matched_signals_raw.npy"
    
#     echo "Results saved in $output_dir"
# done

# echo "=========================================="
# echo "All dictionaries tested!"
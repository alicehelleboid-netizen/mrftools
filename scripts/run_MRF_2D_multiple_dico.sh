set -e

# # bash scripts/run_MRF_2D_multiple_dico.sh <DATA_FILE>

# # Test data filename
DATA_FILE=$1

# # Template configuration files
# CONFIG_TEMPLATES=(
#     "dico/mrf_dictconf_Dico2_Invivo_overshoot_0_55T.json"
#     "dico/mrf_dictconf_Dico2_Invivo_light_for_matching_overshoot_0_55T.json"
# )

# # Parameters to test: array of "wT2,fT2" pairs init (50.0,95.0) et à 3T (40,80)
# PARAM_SETS=(
#     # "50.0,80.0"       
#     "50.0,95.0"      # Default wT2,fT2 0.55T
#     "50.0,130.0"     
#     "50.0,150.0"     
#     "30,95.0"      
#     "70.0,95.0" 
#     "100.0,95.0" 
#     "70.0,130.0" 
#     "30.0,80.0"
#     "100.0,150.0"      
# )

# # Generate dictionaries for each template and parameter set
# echo "=========================================="
# echo "Generating dictionaries with different parameters..."
# echo "=========================================="

# DICOS=()
    
# for params in "${PARAM_SETS[@]}"
# do

#     # Extract template name for naming
#     CONFIG_TEMPLATE=${CONFIG_TEMPLATES[0]}
#     template_base=$(basename "$CONFIG_TEMPLATE" .json)
#     echo ""
#     echo "Processing template: $template_base"
#     echo "----------------------------------------"

#     IFS=',' read -r wT2 fT2 <<< "$params"
#     dico_name="${template_base}_wT2_${wT2}_fT2_${fT2}.json"
#     dico_path="dico/$dico_name"
    
#     echo "Creating dictionary: $dico_name (wT2=$wT2, fT2=$fT2)"
    
#     # Copy template and modify parameters using Python
#     python3 << EOF
# import json

# # Load template
# with open("$CONFIG_TEMPLATE", "r") as f:
#     config = json.load(f)   

# # Update T2 parameters
# config["water_T2"] = float($wT2)
# config["fat_T2"] = float($fT2)

# # Save modified config
# with open("$dico_path", "w") as f:
#     json.dump(config, f, indent=1)

# print(f"  ✓ Config saved to $dico_path")

# EOF

#     IFS=',' read -r wT2 fT2 <<< "$params"
#     dico_name="${template_base}_light_wT2_${wT2}_fT2_${fT2}.json"
#     dico_path_light="dico/$dico_name"
    
#     echo "Creating dictionary: $dico_name (wT2=$wT2, fT2=$fT2)"
    
#     # Extract template name for naming
#     CONFIG_TEMPLATE=${CONFIG_TEMPLATES[1]}
#     template_base=$(basename "$CONFIG_TEMPLATE" .json)
#     echo ""
#     echo "Processing template: $template_base"
#     echo "----------------------------------------"
#     # Copy template and modify parameters using Python
#     python3 << EOF
# import json

# # Load template
# with open("$CONFIG_TEMPLATE", "r") as f:
#     config = json.load(f)

# # Update T2 parameters
# config["water_T2"] = float($wT2)
# config["fat_T2"] = float($fT2)

# # Save modified config
# with open("$dico_path_light", "w") as f:
#     json.dump(config, f, indent=1)

# print(f"  ✓ Config saved to $dico_path_light")
# EOF



#     # Generate dictionary using gen_dict
#     echo "  Generating dictionary with gen_dict..."
#     bash scripts/run_gen_dict.sh "$dico_path" "$dico_path_light"
#     mv mrf_dict/dico_TR1.89_reco5.0.pkl dico/dico_TR1.89_reco5.0_wT2_${wT2}_fT2_${fT2}.pkl
    
#     # Add to DICOS array
#     DICOS+=("mrf_dict/dico_TR1.89_reco5.0_wT2_${wT2}_fT2_${fT2}.pkl")
#     echo "  ✓ Completed for $dico_name"
#     echo ""
# done



# echo "=========================================="
# echo "Dictionary generation complete!"
# echo "Generated ${#DICOS[@]} dictionaries"
# echo "=========================================="

# exit
# 
DICOS=('dico_TR1.89_reco5.0_wT2_30_fT2_95.0' 'dico_TR1.89_reco5.0_wT2_30.0_fT2_80.0' 'dico_TR1.89_reco5.0_wT2_50.0_fT2_130.0' 'dico_TR1.89_reco5.0_wT2_50.0_fT2_150.0' 'dico_TR1.89_reco5.0_wT2_70.0_fT2_130.0' 'dico_TR1.89_reco5.0_wT2_100.0_fT2_95.0' 'dico_TR1.89_reco5.0_wT2_100.0_fT2_150.0' 'dico_TR1.89_reco5.0_wT2_50.0_fT2_95.0' 'dico_TR1.89_reco5.0_wT2_70.0_fT2_95.0')

# echo "=========================================="
# echo "Extracting k-space data..."
# python scripts/cli.py extract_data --filename "$DATA_FILE.dat"

# echo "=========================================="
# echo "Estimating coil sensitivity maps..."
# python scripts/cli.py calculate_sensi --filekdata data/kdata.npy

# echo "=========================================="
# echo "Reconstructing volumes..."
# python scripts/cli.py build_volumes --filekdata data/kdata.npy --fileb1 data/b1.npy

# echo "=========================================="
# echo "Building masks..."
# python scripts/cli.py build_masks --filekdata data/kdata.npy --fileb1 data/b1.npy --threshold-factor 0.06

echo "=========================================="
echo "Starting multiple dictionary testing..."

for dico_file in "${DICOS[@]}"
    
do
    echo ""
    echo "=========================================="
    echo "Testing dictionary: $dico_file"
    echo "=========================================="
    
    # Get dictionary name without extension for output folder
    dico_name="${dico_file}"
    
    # Create output directory for this dictionary
    output_dir="results_$dico_name"
    # mkdir -p "$output_dir"
    
    # # Reconstruct singular volumes for this dictionary
    # echo "Reconstructing singular volumes for $dico_name..."
    # python scripts/cli.py build_volumes_singular --filekdata data/kdata.npy --fileb1 data/b1.npy --dictdir dico --dictfiles "$dico_file.pkl" --L0 6 --niter 1 --regularizer wavelet --lambd 1e-4
    
    # # Copy and rename singular volumes
    # cp data/volumes_singular.npy "$output_dir/volumes_singular.npy"
    
    # # Build masks for singular volumes
    # echo "Building masks for singular volumes..."
    # python scripts/cli.py build_masks_singular --filevolumes "$output_dir/volumes_singular.npy" --l 0 --threshold 0.06
    
    # Copy and rename singular masks
    # cp data/masks_singular.npy "$output_dir/masks_singular.npy"
    
    # Build maps for this dictionary
    echo "Building parameter maps for $dico_name..."
    python scripts/cli.py build_maps --filevolumes "$output_dir/volumes_singular.npy" --filemasks "$output_dir/masks_singular.npy" --fileseq data/dico_seqParams.pkl --dictfile "dico/$dico_file.pkl" --volumestype singular --returncost True
    
    # python scripts/cli.py build_maps --filevolumes data/volumes.npy --filemasks data/masks.npy --fileseq data/dico_seqParams.pkl --dictfiles "dico/$dico_file.pkl" --volumestype raw --returncost True
    
    # Copy all results to output directory
    cp data/maps.pkl "$output_dir/all_maps_singular_v2.pkl.npy"
    cp /home/ahelleboid/Low_field_MRF/mrftools/map_rebuilt.pkl.npy "$output_dir/map_rebuilt_singular_v2.pkl" 
    cp /home/ahelleboid/Low_field_MRF/mrftools/matched_signals_coarse_dico.npy "$output_dir/matched_signals_singular_v2.npy"
    
    echo "Results saved in $output_dir"
done

echo "=========================================="
echo "All dictionaries tested!"
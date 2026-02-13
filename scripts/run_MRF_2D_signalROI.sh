set -e

# # Extracting k-space 
echo "-----------------------------Extracting k-space data..."
python scripts/cli.py extract_data --filename  $1.dat

# # # Coil sensitivity maps estimation
echo "-----------------------------Estimating coil sensitivity maps..."
python scripts/cli.py calculate_sensi --filekdata data/kdata.npy

# # volume reconstruction
echo "-----------------------------Reconstructing volumes..."
python scripts/cli.py build_volumes --filekdata data/kdata.npy --fileb1 data/b1.npy
# python scripts/cli.py build_volumes_singular --filekdata data/kdata.npy --fileb1 data/b1.npy --dictdir . --dictfiles $2 --L0 6 --niter 1 --regularizer wavelet --lambd 1e-4

# # Mask
# echo "-----------------------------Building masks..."
# python scripts/cli.py build_masks_singular --filevolumes data/volumes_singular.npy --l 0 --threshold 0.06
python scripts/cli.py build_masks --filekdata data/kdata.npy --fileb1 data/b1.npy --threshold-factor 0.06

# build maps
echo "-----------------------------Building parameter maps..."
# python scripts/cli.py build_maps --filevolumes data/volumes_singular.npy --filemasks data/masks_singular.npy --fileseq data/dico_seqParams.pkl --dictfile $2 --volumestype singular --returncost True
python scripts/cli.py build_maps --filevolumes data/volumes.npy --filemasks data/masks.npy --fileseq data/dico_seqParams.pkl --dictfiles $2 --signal $3 --volumestype raw --returncost True

# spec=""
# base=$(basename "$f")
# for f in data/*.mha; do
#   base=$(basename "$f" .mha)
#   cp "$f" "/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#5_2025_LF_MRF/3_Data_Processed/low_field/${1#data/}_${base}${spec}.mha"
# done


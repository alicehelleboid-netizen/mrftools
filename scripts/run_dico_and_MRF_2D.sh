set -e


SEQ_ADJ="/home/ahelleboid/Low_field_MRF/mrftools/dico_config/mrf_sequence_adjusted_0_55T_TEv1.json"
DICT_CONFIG="/home/ahelleboid/Low_field_MRF/mrftools/dico_config/water/mrf_dictconf_Dico2_Invivo_overshoot_0_55T_fT1_water.json"
DICT_CONFIG_LIGHT="/home/ahelleboid/Low_field_MRF/mrftools/dico_config/water/mrf_dictconf_Dico2_Invivo_light_for_matching_overshoot_0_55T_fT1_water.json"
DAT_FILE="/home/ahelleboid/Low_field_MRF/mrftools/data/MRF_Dixon/meas_MID00021_FID00517_raFin_FreeMax_recoOK.dat"

DAT_DIR=$(dirname "$DAT_FILE")
echo "$DAT_DIR"
DEST="$DAT_DIR"

# # build dico
# echo "-----------------------------Building dictionary..." 
# python scripts/cli.py generate_dico --sequencefile $1 --dictconf $2 --dictconflight $3 --reco 5.0 --echospacing 1.11 --TI 8.32 --isbuildphi True
# python scripts/cli.py gen_dict_mrf --datafile $DAT_FILE.dat --force True --dest $DEST --wait-time 5.0 --seqfile $SEQ_ADJ --dictconf $DICT_CONFIG --dictconf-light $DICT_CONFIG_LIGHT --isbuildphi True --withff

DICO_PATH="/home/ahelleboid/Low_field_MRF/mrftools/DICO_MRF/dico_TR1.89_reco5.0_full_overshoot.pkl"

echo "-----------------------------Extracting k-space data..."
python scripts/cli.py extract_data --filename $DAT_FILE

echo "-----------------------------Estimating coil sensitivity maps..."
python scripts/cli.py calculate_sensi --filekdata $DAT_DIR/kdata.npy

echo "-----------------------------Reconstructing volumes..."
python scripts/cli.py build_volumes_singular \
  --filekdata $DEST/kdata.npy \
  --fileb1 $DEST/b1.npy \
  --dictdir . \
  --dictfiles $DICO_PATH \
  --L0 6 --niter 1 --regularizer wavelet --lambd 1e-4

echo "-----------------------------Building masks..."
python scripts/cli.py build_masks_singular --filevolumes "$DEST/volumes_singular.npy" --l 0 --threshold 0.06 --it 3
exit
echo "-----------------------------Building parameter maps..."
python scripts/cli.py build_maps \
  --filevolumes $DEST/volumes_singular.npy \
  --filemasks $DEST/masks_singular.npy \
  --fileseq $DEST/dico_seqParams.pkl \
  --dictfile $DICO_PATH \
  --volumestype singular --returncost True --with-FF


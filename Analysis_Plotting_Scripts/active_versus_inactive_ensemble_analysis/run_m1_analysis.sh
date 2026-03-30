#!/usr/bin/env bash
set -euo pipefail

SYSTEM="m1-clemastine"
REPS="replica1 replica2 replica3"

# Residue numbers (your topology)
ASP332=86
TYR333=87
TYR651=386
TYR739=409
TYR753=423

# Reference structures (can be relative to where you run the script)
INACTIVE_REF="m1.pdb"
ACTIVE_REF="9UCP_renum_to_m1.pdb" # this file was aligned with the included script, so residue numbering matches between PDB files

for REP in $REPS; do
    echo "Processing $SYSTEM / $REP"

    BASE="../../Data/M1_muscarinic_receptor/complexes/clemastine/analysis/${REP}"
    OUTDIR="${BASE}/analysis_active_vs_inactive"
    mkdir -p "$OUTDIR"

    # -------------------------
    # Resolve files (exactly one each)
    # -------------------------

    XTC_FILES=("${BASE}"/*.xtc)
    if [ "${#XTC_FILES[@]}" -ne 1 ]; then
        echo "ERROR: expected exactly one .xtc in ${BASE}, found ${#XTC_FILES[@]}"
        printf '  %s\n' "${XTC_FILES[@]}"
        exit 1
    fi
    XTC="${XTC_FILES[0]}"

    TPR_FILES=("${BASE}"/*.tpr)
    if [ "${#TPR_FILES[@]}" -ne 1 ]; then
        echo "ERROR: expected exactly one .tpr in ${BASE}, found ${#TPR_FILES[@]}"
        printf '  %s\n' "${TPR_FILES[@]}"
        exit 1
    fi
    TPR="${TPR_FILES[0]}"

    NDX="index_m1-clemastine.ndx"
   
    # -------------------------
    # Ligand–residue distances (min heavy-atom distance)
    # -------------------------
    printf "1\n2\n" | gmx mindist -s "$TPR" -f "$XTC" -n "$NDX" \
      -od "$OUTDIR/mindist_UNL_r86.xvg" >/dev/null


    printf "1\n3\n" | gmx mindist -s "$TPR" -f "$XTC" -n "$NDX" \
      -od "$OUTDIR/mindist_UNL_r87.xvg" >/dev/null


    printf "1\n4\n" | gmx mindist -s "$TPR" -f "$XTC" -n "$NDX" \
      -od "$OUTDIR/mindist_UNL_r386.xvg" >/dev/null


    printf "1\n5\n" | gmx mindist -s "$TPR" -f "$XTC" -n "$NDX" \
      -od "$OUTDIR/mindist_UNL_r409.xvg" >/dev/null


    printf "1\n6\n" | gmx mindist -s "$TPR" -f "$XTC" -n "$NDX" \
      -od "$OUTDIR/mindist_UNL_r423.xvg" >/dev/null


    # -------------------------
    # TM3–TM6 intracellular opening (COM distance)
    # -------------------------
    gmx distance -s "$TPR" -f "$XTC" -n "$NDX" \
      -select "com of group TM3_IC plus com of group TM6_IC" \
      -oall "$OUTDIR/tm3dic_tm6ic_dist.xvg" >/dev/null

    gmx distance -s "$TPR" -f "$XTC" -n "$NDX" \
      -select "com of group TM3_DRY plus com of group TM6_IC" \
      -oall "$OUTDIR/tm3dry_tm6ic_dist.xvg" >/dev/null

    gmx distance -s "$TPR" -f "$XTC" -n "$NDX" \
      -select "com of group TM3_r104 plus com of group TM6_r365" \
      -oall "$OUTDIR/tm3r104_tm6r365_dist.xvg" >/dev/null

    # -------------------------
    # RMSD to inactive and active references of the TM-CORE region
    # (You will be prompted to choose groups unless we supply -fit/-sel.
    # For now: interactively select e.g. "C-alpha" for both.
    # -------------------------
    # echo "NOTE: gmx rms will prompt you to select groups (choose C-alpha)."
    # printf "21\n21\n" | gmx rms -s "$INACTIVE_REF" -f "$XTC" -n "$NDX" \
    #   -o "$OUTDIR/rmsd_to_inactive.xvg"

    # printf "21\n21\n" | gmx rms -s "$ACTIVE_REF" -f "$XTC" -n "$NDX" \
    #   -o "$OUTDIR/rmsd_to_active.xvg"

    # echo "Done: $SYSTEM / $REP -> $OUTDIR"

    #-------------------------

    # Calculate dihedral angle of Tyr^7.53 (Tyr423)
    # this can be compared to the dihedral angle of Tyr^7.53 in the active (9UCP) and inactive crystal structure (5CXV)
    # Example for gromacs2022 (interactive once; then you can pipe)
    printf "26\n" | gmx angle -f "$XTC" -n index_m1-clemastine.ndx \
    -type dihedral \
    -ov "$OUTDIR/chi1_Tyr423.xvg" -all


done

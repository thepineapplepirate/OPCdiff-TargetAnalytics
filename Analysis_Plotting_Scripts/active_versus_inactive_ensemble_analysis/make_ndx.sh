
cat > make_ndx_cmds.txt << 'EOF'
keep 0
r UNL
r 86
r 87
r 386
r 409
r 423
r 96-111
r 330-344
name 1 UNL
name 2 r_86
name 3 r_87
name 4 r_386
name 5 r_409
name 6 r_423
name 7 TM3_IC
name 8 TM6_IC
q
EOF

gmx make_ndx -f tprfile.tpr -o index_required.ndx < make_ndx_cmds.txt | index.log


#!/usr/bin/env python3
from pathlib import Path


# This is an auxiliary script that will align and renumber pdb files based sequence overlap.

aa3to1 = {
 "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G","HIS":"H",
 "ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W",
 "TYR":"Y","VAL":"V"
}

def read_pdb_seq(pdb_path):
    lines = Path(pdb_path).read_text().splitlines()
    residues = []
    seen = set()
    for ln in lines:
        if ln.startswith("ATOM"):
            resn = ln[17:20].strip()
            chain = ln[21].strip()
            resid = int(ln[22:26])
            key = (chain, resid)
            if key not in seen:
                seen.add(key)
                residues.append((chain, resid, resn))
    seq = "".join(aa3to1.get(r[2], "X") for r in residues)
    return residues, seq, lines

def needleman_wunsch(a, b, match=1, mismatch=-1, gap=-1):
    # Simple global alignment
    n, m = len(a), len(b)
    score = [[0]*(m+1) for _ in range(n+1)]
    bt = [[None]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        score[i][0] = score[i-1][0] + gap; bt[i][0] = "U"
    for j in range(1, m+1):
        score[0][j] = score[0][j-1] + gap; bt[0][j] = "L"
    for i in range(1, n+1):
        for j in range(1, m+1):
            sdiag = score[i-1][j-1] + (match if a[i-1]==b[j-1] else mismatch)
            sup   = score[i-1][j] + gap
            sleft = score[i][j-1] + gap
            best = max(sdiag, sup, sleft)
            score[i][j] = best
            bt[i][j] = "D" if best==sdiag else ("U" if best==sup else "L")
    # traceback
    i, j = n, m
    aa, bb = [], []
    while i>0 or j>0:
        step = bt[i][j]
        if step == "D":
            aa.append(a[i-1]); bb.append(b[j-1]); i-=1; j-=1
        elif step == "U":
            aa.append(a[i-1]); bb.append("-"); i-=1
        else:
            aa.append("-"); bb.append(b[j-1]); j-=1
    return "".join(reversed(aa)), "".join(reversed(bb))

def renumber_9ucp_to_m1(m1_pdb, ucp_pdb, out_pdb):
    m1_res, m1_seq, _ = read_pdb_seq(m1_pdb)
    u_res,  u_seq,  u_lines = read_pdb_seq(ucp_pdb)

    aln_m1, aln_u = needleman_wunsch(m1_seq, u_seq)

    # Build mapping: (u_chain,u_resid) -> m1_resid
    mapping = {}
    m1_i = 0
    u_i  = 0
    for am, au in zip(aln_m1, aln_u):
        if am != "-": m1_i += 1
        if au != "-": u_i  += 1
        if am != "-" and au != "-":
            m1_resid = m1_res[m1_i-1][1]
            u_key = (u_res[u_i-1][0], u_res[u_i-1][1])
            mapping[u_key] = m1_resid

    # Rewrite PDB with renumbered residues where mapping exists
    out = []
    for ln in u_lines:
        if ln.startswith("ATOM"):
            chain = ln[21].strip()
            resid = int(ln[22:26])
            key = (chain, resid)
            if key in mapping:
                new_resid = mapping[key]
                ln = ln[:22] + f"{new_resid:4d}" + ln[26:]
        out.append(ln)
    Path(out_pdb).write_text("\n".join(out) + "\n")
    print(f"Wrote: {out_pdb}")

if __name__ == "__main__":

    renumber_9ucp_to_m1("m1.pdb", "9UCP_clean.pdb", "9UCP_renum_to_m1.pdb")

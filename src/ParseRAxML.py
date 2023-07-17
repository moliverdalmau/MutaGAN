'''
​​© 2020-2022 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

This material may be only be used, modified, or reproduced by or for the U.S. Government pursuant to the license rights granted under the clauses at DFARS 252.227-7013/7014 or FAR 52.227-14. For any other permission, please contact the Office of Technology Transfer at JHU/APL.
'''


import os
import re
import sys
import subprocess
import argparse
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq

def parse_RAxML_w_R(nodeLabelfilename: str, edgefilename: str, nodefilename: str, tipfilename: str):
    '''
    Code for parsing and extracting the parent child pairs of an RAxML phylogenetic tree.
    inputs:
        nodeLabelfilename: contains the data from RAxML
        writes data to: edgefilename
                        nodefilename
                        tipfilename
    '''
    cmd=['Rscript'] +['ParseRAxML.R'] + ["--filename", nodeLabelfilename]
    if edgefilename is not None:
        cmd +=["--edgefilename", edgefilename]
    if nodefilename is not None:
        cmd += ['--nodefilename', nodefilename]
    if tipfilename is not None:
        cmd += ['--tipfilename', tipfilename]
    
    print(cmd)
    try:
        output = subprocess.run(cmd)
        if (output.returncode != 0):
            raise OSError(f"Rscript returned an error: {output.returncode}")
    except FileNotFoundError:
        raise OSError("Rscript not found - please make sure you have R installed")

def parse_tree(marginalAncestralfilename: str, tipfilename: str, alignmentfilename: str, edgefilename: str, coding_sequence_start: int, coding_sequence_end: int, outputfilename: str):
    # node sequeces generated by RAxML, does not include tip sequences
    seq = pd.read_csv(marginalAncestralfilename, sep=' ', header=None)
    # This file is generated by R 
    tip_labels = pd.read_csv(tipfilename, header=0, names=['tip_index', 'record_id'])

    # Alignment file generated by mafft - includes only aligned tip sequences
    tip_sequences = list(SeqIO.parse(alignmentfilename, format='fasta'))

    # this file generated by R 
    indices = pd.read_csv(edgefilename)

    # provide column names for ease of use with pandas
    seq.rename(columns = {0:'index', 1:'sequence'}, inplace=True)

    # Cleaned sequences by removing illegal characters. 
    seq.loc[:, 'sequence'] = seq.loc[:, 'sequence'].apply(lambda x: x[int(coding_sequence_start):int(coding_sequence_end)])
    ##seq.loc[:, 'sequence'] = seq.loc[:, 'sequence'].apply(lambda x: x[int(coding_sequence_start):int(coding_sequence_end)] if isinstance(x, str) else x)


    # translate nucleotides to amino acids
    seq.loc[:,'sequence'] = seq.loc[:,'sequence'].apply(lambda x: str(Seq(x).translate()[:-1]))
    ##seq.loc[:,'sequence'] = seq.loc[:,'sequence'].apply(lambda x: str(Seq(str(x)).translate()[:-1]))


    # change root index label to the last row
    newidx=str(int(seq['index'].loc[~(seq['index']=='ROOT')].max())+1)

    seq.loc[seq['index']=='ROOT',"index"]= newidx

    # Translate nucleotide sequences to amino acids for tip sequences
    tip_sequences_new = []
    tip_sequences_header = []
    for record in tip_sequences:
        tip_sequences_new.append(str(''.join(record.seq[coding_sequence_start:coding_sequence_end].translate()[:-1])))
        tip_sequences_header.append(record.id)

    # Turn fasta records into a dataframe
    tip_sequences_frame = pd.DataFrame(list(zip(tip_sequences_header, tip_sequences_new)), columns=['record_id', 'sequence_id'])

    # Assign tip labels / sequences to a tree index
    merge_tips = tip_sequences_frame.merge(tip_labels, left_on='record_id', right_on='record_id', how='outer')

    # Rename columns merge_tips to match seq object columns
    merge_tips.rename(columns={'sequence_id':'sequence', 'tip_index':'index'}, inplace=True)

    # Concatenating the tree index - sequence for nodes and tree index - sequence for tips
    nodes_tips = pd.concat([merge_tips[['index', 'sequence']], seq[['index', 'sequence']]], axis=0)

    # Convert nodes_tips index to a string
    nodes_tips['index'] = nodes_tips['index'].astype('str')

    # Remove unnecessary first column
    indices.drop('Unnamed: 0', axis=1, inplace=True)

    # Rename columns
    indices.rename(columns={'V1':'ancestor', 'V2':'descendant'}, inplace=True)

    # Convert index to string
    indices[['ancestor', 'descendant']] = indices[['ancestor', 'descendant']].astype('str')

    # Merge ancestor and descendant node indices with node and tip sequences
    ancestor = indices.merge(nodes_tips, left_on='ancestor', right_on='index', how='left')
    descendant = indices.merge(nodes_tips, left_on='descendant', right_on='index', how='left')

    # Combine into final format for input into mutagan
    final = pd.concat([ancestor[['ancestor','sequence']],descendant[['descendant','sequence']]], axis=1)
    final.to_csv(outputfilename, index=None)

def main(arguments):
    '''
    The inputs are:
        nodeLabelfilename: an output file from RAxML
        marginalAncestralfilename: an output from RAxML
        aln.fasta: an output from Mafft
        edgefilename: the edge file from parsing the trees
        nodefilename: the node file from parsing the trees
        tipfilename: the edge file from parsing the trees
        coding_sequence_start: an integer for the start of the sequences. Count like a biologists so first index is 1, rather than 0
        coding_sequence_end: an integer for the end of the sequences. Count like a biologists so first index is 1, rather than 0
        outputfilename: the filename for where the parent-child pairs should be saved to
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodeLabelfilename", default="../data/input_model/RAxML_nodeLabelledRootedTree.RAxML_reroot.ancestral", 
              help="RAxML filename for parsing")
    parser.add_argument("--marginalAncestralfilename", default="../data/input_model/RAxML_marginalAncestralStates.RAxML_reroot.ancestral", 
              help="RAxML filename for parsing")
    parser.add_argument("--alignmentfilename", default= "../data/input_model/aln.fasta", help="alignment file generated by mafft - inludes only aligned tip sequences")
    parser.add_argument("--edgefilename", default="../data/input_model/RAxML_ancestor_index.csv", 
              help="edge output file name")
    parser.add_argument("--nodefilename", default="../data/input_model/RAxML_ancestor_node_label.csv", help="node output file name")
    parser.add_argument("--tipfilename", default="../data/input_model/tip_labels.csv", 
              help="tip output file name")
    parser.add_argument("--coding_sequence_start", default= 29, type=int, help="Start of coding sequence for influenza")
    parser.add_argument("--coding_sequence_end", default= 1730, type=int, help="Start of coding sequence for influenza")
    parser.add_argument("--outputfilename", default= '../data/input_model/test_set_mutagan_2018_2019.csv', help="Start of coding sequence for influenza")

    args = parser.parse_args(arguments)
    coding_sequence_start = args.coding_sequence_start-1 

    if not os.path.exists(args.nodeLabelfilename):
        raise ValueError(f"The filename {args.nodeLabelfilename} does not exist. Please make sure it exists.")

    if not os.path.exists(args.marginalAncestralfilename):
        raise ValueError(f"The filename {args.marginalAncestralfilename} does not exist. Please make sure it exists.")

    if not os.path.exists(args.alignmentfilename):
        raise ValueError(f"The filename {args.alignmentfilename} does not exist. Please make sure it exists.")

    parse_RAxML_w_R(args.nodeLabelfilename, args.edgefilename, args.nodefilename, args.tipfilename)
    parse_tree(args.marginalAncestralfilename, args.tipfilename, args.alignmentfilename, args.edgefilename, args.coding_sequence_start, args.coding_sequence_end, args.outputfilename)

if __name__=="__main__":
    main(sys.argv[1:])
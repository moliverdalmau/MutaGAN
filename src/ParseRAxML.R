## '''
## © 2020-2022 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).
##
## This material may be only be used, modified, or reproduced by or for the U.S. Government pursuant to the license rights granted under the clauses at DFARS 252.227-7013/7014 or FAR 52.227-14. For any other permission, please contact the Office of Technology Transfer at JHU/APL.
## '''

list.of.packages <- c("ape","optparse")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages,repos='http://cran.us.r-project.org')

library(ape)
library(optparse)

args = commandArgs(trailingOnly=TRUE)

parseRAxMLFile <- function(filename,edgefilename, nodefilename, tipfilename){
    mut = read.tree(filename)
    write.csv(mut$edge, edgefilename)
    write.csv(mut$node.label, nodefilename)
    write.csv(mut$tip.label, tipfilename)
}


option_list = list(
    make_option("--filename", type="character", default="RAxML_marginalAncestralStates.RAxML_reroot.ancestral", 
              help="RAxML filename for parsing", metavar="character"),
    make_option("--edgefilename", type="character", default="RAxML_ancestor_index.csv", 
              help="edge output file name", metavar="character"),
    make_option("--nodefilename", type="character", default="RAxML_ancestor_node_label.csv", 
              help="node output file name", metavar="character"),
    make_option("--tipfilename", type="character", default="tip_labels.csv", 
              help="tip output file name", metavar="character")
); 
 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

parseRAxMLFile(opt$filename,opt$edgefilename,opt$nodefilename, opt$tipfilename)


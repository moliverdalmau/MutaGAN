# MutaGAN

## Requirements

We recommend that you have least one GPU to run this code.  Without a GPU, you will experience long run times and high memory usage.

We used the phylogenetic software Randomized Axelerated Maximum Likelihood (RAxML) to create our phylogenetic trees and mafft to produce an alignment file. Our code for parsing those trees is geared towards the output from RAxML.

### Environment Setup

#### R

This project requires R version 4.1.2.

Use the following instructions to install Rscript and ensure it runs from the command line/terminal:

1. Go to the R-project page and download from the appropriate mirror: <https://cran.r-project.org/mirrors.html>
2. On Windows, you might need to add R to the environment path. This means adding the location where R was installed (usually `C:\Program Files\R\R-4.1.2\bin\i386`) to the PATH.  
  For Linux or Macs, this should only be necessary if you installed Rstudio without installing base R first. The easiest solution is to install base R but alternatively, you can add the location to your path using `export PATH="<path/to/R>":$PATH`
3. To check if this works, open a new command window or terminal and try running Rscript and it should NOT say _Rscript command could not be found_

The first time that `ParseRAxML.py` or `ParseRAxML.R` is run, the project will automatically download two required libraries, `ape` and `optparse`, from the default <http://cran.us.r-project.org> repository.

#### Python

MutaGAN was tested on Python 3.8.8 that was installed using Anaconda. Package requirements are found in `requirements.txt` and can be installed using the following command:

```sh
pip install -r requirements.txt
```

### PyMOL

PyMOL is only required if you need to create
PyMOL is used to create the 3D protein models.  It is not required to run and train the model, so it is not included in the `requirements.txt` dependencies.  This section provides instructions to install PyMOL.  If you do not need to create protein models, you may skip this section.

For Windows users, run the following command from the MutaGAN directory to install the PyMOL wheel file:

```sh
pip install whl/pymol-2.5.0-cp38-cp38-win_amd64.whl
```

This Windows wheel file was obtained from <https://www.lfd.uci.edu/~gohlke/pythonlibs/#pymol>.

For Linux or Mac, you can install PyMOL using Conda:

```sh
conda install -c schrodinger -c conda-forge pymol
```

If you need more than just the PyMOL python package, use the instructions at <https://pymol.org/2/support.html?#installation> to install the entire PyMOL software package.

## Running the code

Unzip the `data.zip` and `utilities.zip` archives - they will unpack into a `data` and `utilities` folder, respectively.

If you need to convert RAxML trees into CSVs of parents and children, run the provided `ParseRAxML.py` script from the `src` folder:

```sh
python ParseRAxML.py
```

To run the MutaGAN code from a command line, run:

```sh
python BoosterShot_Influenza_MutaGAN.py
```

## Files

The following files are included in this repository:

**`/src` folder:**

* `BoosterShot_Influenza_MutaGAN.py`:  The main function that is used to run the code, train the model, and run the validation. This script uses a hardcoded set of parameters that you may change as you see fit.
* `boostershot_utility_functions.py`: A file containing the functions needed to run the main file
* `H3_pymol_4gms20191204.py`: A file for generating the PyMol figures
* `ParseRAxML.py`: Python code for converting the outputs of RAxML (the marginal ancesteral states, and the node labeled rooted tree) and the mafft produced alignment file into pairs of parent-child pairs.  This script runs `ParseRAxML.R`.
* `ParseRAxML.R`: An R function that quickly executes part of ParseRAxML. When you run this function using R or through `ParseRAxML.py`, R will automatically install the `ape` and `optparse` packages if it hasn't done so already.

**`data.zip` archive:**

* `4gms.pdb`: file for plotting the 3D protein
* `20190220_IG_influenza_GAN_test_weighted_paths_and_leaves_testUniqueParentsDiffParentChild.csv`: testing data where the parent and child are different
* `20190220_IG_influenza_GAN_test_weighted_paths_and_leaves_testUniqueParentsSameParentChild.csv`: testing data where the parent and child are the same
* `20190220_IG_influenza_GAN_test_weighted_paths_and_leaves_trainUniqueParentsDiffParentChild.csv`: training data where the parent and child are different
* `20190220_IG_influenza_GAN_test_weighted_paths_and_leaves_trainUniqueParentsSameParentChild.csv`: training data where the parent and child are the same
* `test_set_mutagan_2018_2019.csv`: Validation data
* `input_model/RAxML_nodeLabelledRootedTree.RAxML_reroot.ancestral`: node-labelled rooted tree output from the building a tree in RAxML using 2018-2019 data
* `input_model/RAxML_marginalAncestralStates.RAxML_reroot.ancestral`: marginal ancestral states output from the building a tree in RAxML using 2018-2019 data
* `input_model/aln.fasta`: mafft alignment file

**`utilities.zip` archive:**

* `BadDecoder_ForBadDataGeneration.h5`: A failed decoder model for generation of bad data
* `BADDecoderFromEarlyModel4500_250.json`: A failed decoder model architecture for bad data generation
* `BadEncoder_ForBadDataGeneration.h5`: A failed encoder model for generation of bad data
* `BADEncoderFromEarlyModel4500_250.json`: A failed encoder model architecture for bad data generation
* `influenzaFormatting.csv`: proper formatting for HA influenza protein for easy interpretation
* `Sneath Index Similarity.csv`: Sneath Similarity index matrix
* `TokenizerGANV3.5.pkl`: a pickled version of the tokenizer for training on 4500 tokens
* `unmatchingsequenceschild.npy`: The unmatching children for model training
* `unmatchingsequencesparent.npy`: The unmatching parents for model training
* `unmatchingsequencesdiff.npy`: The Levenshtein distance between the parents and the children
* `Influenza_biLSTM_encoder_model_128_4500_weightsV3.h5`: a perviously trained encoder used to seed the weight sharing. These values get changed later so it's only there for weight sharing purposes.

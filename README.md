# MutantX-S

MutantX-S is a static malware classification system. It employs techniques such as feature hashing and prototype-based clustering to conserve computational costs, giving the
system the ability to scale up to large datasets. This work is our rendition of MutantX-S in Python, the system is created using the logic and algorithms outlined in the following
paper:

* X. Hu, S. Bhatkar, K. Griffin, and K. G. Shin. 2013. MutantX-S: Scalable
Malware Clustering Based on Static Features. In Proceedings of the 2013
USENIX Conference on Annual Technical Conference (San Jose, CA)
(USENIX ATC’13). USENIX Association, USA, 187–198.

The goal of this work is to allow for benchmarking between malware classifiers. The original work was done using extracted opcodes as features, this data was taken from a private 
dataset. For our rendition, the open source [EMBER](https://github.com/endgameinc/ember) dataset was used, with function imports being used as features, using an open-source 
dataset ensures that any experiments done on the system are reproducible.

## Usage

To get the system up and running, JSON files of malware data must first be downloaded from [EMBER](https://github.com/endgameinc/ember). Each sample from these files will contain a
corresponding MD5. The user must supply the system with a file listing the MD5s representing which malware samples they wish to cluster (with one on each line). These files, along
with the chosen parameters, can be passed to MutantX-S as command line arguments in the following order:

```bash
n_gram_size p_max d_min md5_file json_files
```

with the JSON files each being separated by a space. For more on n_gram_size, p_max and d_min, see the Hu et Al. paper cited above. If nothing is passed via the command line, the
user will be prompted to enter each individually.

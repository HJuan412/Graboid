#!/bin/bash

# create out dir (if needed)
mkdir $1
# retrieve taxdmp file
wget -O $1/taxdmp.zip -t 3 https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdmp.zip

# extract relevant files
unzip -d $1 $1/taxdmp.zip nodes.dmp names.dmp

# Edit files
sed -i 's/\t|//g' $1/nodes.dmp
sed -i 's/\t|//g' $1/names.dmp

# retrieve important columns
grep 'scientific name' $1/names.dmp | cut -f 1,2 > $1/names.tsv
cut -f 1,2,3 $1/nodes.dmp > $1/nodes.tsv
rm $1/nodes.dmp
rm $1/names.dmp
rm $1/taxdmp.zip
exit

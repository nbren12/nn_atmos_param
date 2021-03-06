#!/bin/bash
#
# An example hook script to verify what is about to be committed.
# Called by "git commit" with no arguments.  The hook should
# exit with non-zero status after issuing an appropriate message if
# it wants to stop the commit.
#
# To enable this hook, rename this file to "pre-commit".

# see if plots was edited in the last commit

project_name=uw-machine-learning
report_server=olympus
public_url=https://atmos.washington.edu/~nbren12/reports/$project_name
report_dir=/home/disk/eos4/nbren12/public_html/reports/$project_name

for notebook in $@
do
    ext=${notebook##*.}
    namenoext=${notebook%.*}
    if [ $ext == "ipynb" ]; then
        jupyter nbconvert $notebook
        scp $namenoext.html $report_server:$report_dir/

        url=$public_url/$(basename $namenoext).html
        echo "File URL: $url"
    fi
done

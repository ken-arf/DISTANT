#!/bin/bash


files='./results/*.f1'

output="f1_result.csv"

echo "precision,recall,f1-score,ratio,fname" > $output

for file in $files; do
    echo $file
    cat $file | grep micro > tmp

    echo "raw data:"
    cat tmp

    buf=`cat tmp | cut -c19-43 | sed 's/     /,/g'`

    echo "extractd data: $buf"
    
    fname=`basename $file`


    if [[ $fname =~ "annotate.It_W6" ]]; then
        p=`echo $fname | cut -c27`
        p="0.$p"
        echo $p
    fi

    if [[ $fname =~ "annotate.Scratch_W6" ]]; then
        p=`echo $fname | cut -c22-24`
        echo $p
    fi

    if [[ $fname =~ "annotate.W6" ]]; then
        p=`echo $fname | cut -c14-16`
        echo $p
    fi

    
    echo "$buf,$p,$fname" >> $output
    echo "--"

done

rm tmp

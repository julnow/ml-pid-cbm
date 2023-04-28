#!/bin/bash

#validation of single models
for dir in model_*
    do
        if [[ -d "$dir" ]]; then
            pdfunite $dir/*.pdf $dir.pdf 
       fi
done

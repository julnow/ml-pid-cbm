#!/bin/bash

#validation of single models
mkdir -p pdfs
for dir in model_*
    do
        if [[ -d "$dir" ]]; then
            mkdir -p $dir/merge
            cd $dir
            pdfjam confusion_*.pdf --nup 2x1 --landscape --outfile merge/cm.pdf
            pdfjam correlat*.pdf --nup 2x2 --landscape --outfile merge/corr.pdf
            pdfjam eff_pT*.pdf plot_pt*.pdf --landscape --nup 4x2  --outfile merge/pT.pdf
            pdfjam shap_*.pdf --nup 2x2 --landscape --outfile merge/shap.pdf
            cp roc_plot.pdf merge/roc_plot.pdf
            pdfjam tof_plot_*.pdf --nup 4x3 --landscape --outfile merge/tof.pdf
            cp vars_dist*.pdf merge/
            cd ../  
            pdfunite $dir/merge/*.pdf pdfs/$dir.pdf 
       fi
done
pdfunite all_models/*.pdf pdfs/all_models.pdf

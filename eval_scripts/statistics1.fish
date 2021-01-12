#!/usr/bin/env fish
set MODEL 'ground truth' 'vanila_vae_tf' 'vae_velocR_f0001_t01_trj10_rela' 'vae_velocS_f0001_t01_trj10_rela' 'vanilla_vae_lie_mse_kld01' 'motion_gan' 'conditionedRNN' 'deep_completion'
set FILE 'final_evaluation_mocap_veloc_label3.log'
for model in $MODEL
    echo "===" $model "==="

    grep -e "---> \[$model\] Accuracy:" $FILE | cut -d: -f2 | awk '{n+=1;s+=$1;ss+=$1*$1} END {print "Accuracy:\t[", 1.96 * sqrt((ss - n*(s/n)*(s/n))/n/n), ",", s/n + 1.96 * sqrt((ss - n*(s/n)*(s/n))/n/n), "]"}'

    grep -e "---> \[$model\] FID:" $FILE | cut -d: -f2 | awk '{n+=1;s+=$1;ss+=$1*$1} END {print "FID:\t\t[", 1.96 * sqrt((ss - n*(s/n)*(s/n))/n/n), ",", s/n + 1.96 * sqrt((ss - n*(s/n)*(s/n))/n/n), "]"}'

    grep -e "---> \[$model\] Diversity: " $FILE | cut -d: -f2 | awk '{n+=1;s+=$1;ss+=$1*$1} END {print "Diversity:\t[", 1.96 * sqrt((ss - n*(s/n)*(s/n))/n/n), ",", s/n + 1.96 * sqrt((ss - n*(s/n)*(s/n))/n/n), "]"}'

    grep -e "---> \[$model\] Multimodality: " $FILE | cut -d: -f2 | awk '{n+=1;s+=$1;ss+=$1*$1} END {print "Multimodality:\t[", 1.96 * sqrt((ss - n*(s/n)*(s/n))/n/n), ",", s/n + 1.96 * sqrt((ss - n*(s/n)*(s/n))/n/n), "]"}'
end

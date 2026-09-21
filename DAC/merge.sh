# 生成带 Type 列的汇总文件（推荐，可区分方向）
echo "Dataset,Type,MaskRatio,R@1,R@5,R@10,R@top1,AP" > final_report.csv
awk -F ' *\\| *' 'NR>1 {print $1",D2S,"$2","$3","$4","$5","$6","$7}' dac_summary.txt >> final_report.csv
awk -F ' *\\| *' 'NR>1 {print $1",S2D,"$2","$3","$4","$5","$6","$7}' dac_summary_s2d.txt >> final_report.csv
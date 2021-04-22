from pandas_plink import read_plink1_bin
import vcf


vcf_reader = vcf.Reader(open('../../../data/002_S_0413_SNPs.vcf', 'r'))
for record in vcf_reader:
    print(record)

######### ADNI 1 #########
bed_path_1 = \
'E:/UNC-CS-Course/COMP 790-166/project/multimp/data/ADNI/genetic/ADNI1 GWAS/ADNI_1_GWAS_Plink/WGS_Omni25_BIN_wo_ConsentsIssuesbed'
bim_path_1 = \
'E:/UNC-CS-Course/COMP 790-166/project/multimp/data/ADNI/genetic/ADNI1 GWAS/ADNI_1_GWAS_Plink/ADNI_cluster_01_forward_757LONI.bim'
fam_path_1 = \
'E:/UNC-CS-Course/COMP 790-166/project/multimp/data/ADNI/genetic/ADNI1 GWAS/ADNI_1_GWAS_Plink/ADNI_cluster_01_forward_757LONI.fam'
G1 = read_plink1_bin(bed_path_1, bim_path_1, fam_path_1, verbose=False)
###########################
'''
######### ADNI 3 #########
bed_path_2_1 = \
'E:/UNC-CS-Course/COMP 790-166/project/multimp/data/ADNI/genetic/ADNI3 GWAS/PLINK_Final/ADNI3_PLINK_Final.bed'
bim_path_2_1 = \
'E:/UNC-CS-Course/COMP 790-166/project/multimp/data/ADNI/genetic/ADNI3 GWAS/PLINK_Final/ADNI3_PLINK_Final.bim'
fam_path_2_1 = \
'E:/UNC-CS-Course/COMP 790-166/project/multimp/data/ADNI/genetic/ADNI3 GWAS/PLINK_Final/ADNI3_PLINK_Final.fam'
G3 = read_plink1_bin(bed_path_1, bim_path_1, fam_path_1, verbose=False)
###########################
'''


######### ADNI 3 #########
bed_path_3 = \
'E:/UNC-CS-Course/COMP 790-166/project/multimp/data/ADNI/genetic/ADNI3 GWAS/PLINK_Final/ADNI3_PLINK_Final.bed'
bim_path_3 = \
'E:/UNC-CS-Course/COMP 790-166/project/multimp/data/ADNI/genetic/ADNI3 GWAS/PLINK_Final/ADNI3_PLINK_Final.bim'
fam_path_3 = \
'E:/UNC-CS-Course/COMP 790-166/project/multimp/data/ADNI/genetic/ADNI3 GWAS/PLINK_Final/ADNI3_PLINK_Final.fam'
G3 = read_plink1_bin(bed_path_3, bim_path_3, fam_path_3, verbose=False)
###########################



######### ADNI Omni #########
bed_path_Omni = \
'E:/UNC-CS-Course/COMP 790-166/project/multimp/data/ADNI/genetic/ADNI WGS + Omni2.5M/WGS_Omni2.5M_20140220/WGS_Omni25_BIN_wo_ConsentsIssues.bed'
bim_path_Omni = \
'E:/UNC-CS-Course/COMP 790-166/project/multimp/data/ADNI/genetic/ADNI WGS + Omni2.5M/WGS_Omni2.5M_20140220/WGS_Omni25_BIN_wo_ConsentsIssues.bim'
fam_path_Omni = \
'E:/UNC-CS-Course/COMP 790-166/project/multimp/data/ADNI/genetic/ADNI WGS + Omni2.5M/WGS_Omni2.5M_20140220/WGS_Omni25_BIN_wo_ConsentsIssues.fam'
G_Omini = read_plink1_bin(bed_path_Omni, bim_path_Omni, fam_path_Omni, verbose=False)
###########################
print('1')
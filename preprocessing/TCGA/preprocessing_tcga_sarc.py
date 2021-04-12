import pandas as pd
import numpy as np

#
ROOT='.'
VIEWS = ['Clinical',
         'Methylation (CpG-site level, HM450K)',
         'Methylation (Gene level, HM450K)',
         'Mutation (Gene level)',
         'Mutation (Site level)',
         'RPPA (Analyte Level)',
         'RPPA (Gene Level)',
         'RNAseq (HiSeq, Gene level)',
         'SCNV (Focal level, log-ratio)',
         'SCNV (Focal level, Thresholded)',
         'SCNV (Gene level, log-ratio)',
         'SCNV (Gene level, Thresholded)',]

a = pd.read_table('E:/UNC-CS-Course/COMP 790-166/project/multimp/data/TCGA SARC/Clinical/Human__TCGA_SARC__MS__Clinical__Clinical__01_28_2016__BI__Clinical__Firehose (1).tsi')
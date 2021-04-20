
## Read in raw data ##
### Clinical data
clinical = read.table("/Users/jieun/Dropbox/Desktop/UNC/Spring2021/COMP790/FinalProject/data/tcga-brca/Human__TCGA_BRCA__MS__Clinical__Clinical__01_28_2016__BI__Clinical__Firehose.tsi", header = T) 
ind = apply(clinical, 1, function(x) sum(is.na(x))/length(x)*100) < 50 ## filter only variables with less than 50% missingness
clinical = clinical[ind,] 

rownames(clinical) = clinical$attrib_name # attribution names as rownames
clinical = clinical[,-1] # remove attribution names
clinical = clinical[rownames(clinical) !="overallsurvival",] ## remove overallsurvival (duplicate info with overall_survival)

### RNAseq
rna = read.table("/Users/jieun/Dropbox/Desktop/UNC/Spring2021/COMP790/FinalProject/data/tcga-brca/Human__TCGA_BRCA__UNC__RNAseq__HiSeq_RNA__01_28_2016__BI__Gene__Firehose_RSEM_log2.cct", header=T) 
rownames(rna) = rna$attrib_name
rna = rna[,-1] # remove attribution names
View(rna)
### Mutation - gene level
mutation = read.table("/Users/jieun/Dropbox/Desktop/UNC/Spring2021/COMP790/FinalProject/data/tcga-brca/Human__TCGA_BRCA__WUSM__Mutation__GAIIx__01_28_2016__BI__Gene__Firehose_MutSig2CV.cbt", header=T)
rownames(mutation) = mutation$attrib_name
mutation = mutation[,-1] # remove attribution names

### DNA Methylation
meth = read.table("/Users/jieun/Dropbox/Desktop/UNC/Spring2021/COMP790/FinalProject/data/tcga-brca/Human__TCGA_BRCA__JHU_USC__Methylation__Meth450__01_28_2016__BI__Gene__Firehose_Methylation_Prepocessor.cct", header=T)
rownames(meth) = meth$attrib_name
meth = meth[,-1] # remove attribution names

### Copy Number Variation (SCNV) - gene
copynum_gene = read.table("/Users/jieun/Dropbox/Desktop/UNC/Spring2021/COMP790/FinalProject/data/tcga-brca/Human__TCGA_BRCA__BI__SCNA__SNP_6.0__01_28_2016__BI__Gene__Firehose_GISTIC2.cct", header=T)
rownames(copynum_gene) = copynum_gene$attrib_name
copynum_gene = copynum_gene[,-1] # remove attribution names

### Copy Number Variation (SCNV) - focal
copynum_focal = read.table("/Users/jieun/Dropbox/Desktop/UNC/Spring2021/COMP790/FinalProject/data/tcga-brca/Human__TCGA_BRCA__BI__SCNA__SNP_6.0__01_28_2016__BI__Focal__Firehose_GISTIC2.cct", header=T)
rownames(copynum_focal) = copynum_focal$attrib_name
copynum_focal = copynum_focal[,-1]

### microRNA
microRNA = read.table("/Users/jieun/Dropbox/Desktop/UNC/Spring2021/COMP790/FinalProject/data/tcga-brca/Human__TCGA_BRCA__BDGSC__miRNASeq__HS_miR__01_28_2016__BI__Gene__Firehose_RPKM_log2.cct", header=T)
rownames(microRNA) = microRNA$attrib_name
microRNA = microRNA[,-1] # remove attribution names


## subset samples existing in all views ##
sample <- Reduce(intersect, list(colnames(clinical),colnames(rna),colnames(mutation),
                                 colnames(meth),colnames(microRNA), colnames(copynum_gene), colnames(copynum_focal))) ## 513 samples

clinical_BRCA <- clinical[,sample]
rna_BRCA <- rna[,sample]
mutation_BRCA <- mutation[,sample]
methylation_BRCA <- meth[,sample]
cnv_gene_BRCA <- copynum_gene[,sample]
cnv_focal_BRCA <- copynum_focal[,sample]
miRNA_BRCA <- microRNA[,sample]

## check missingness
### <20%
clinical_miss <- vapply(seq_len(nrow(clinical_BRCA)), function(i) sum(is.na(clinical_BRCA[i,]))*100/ncol(clinical_BRCA),numeric(1))
max(clinical_miss)

### <7%
methylation_miss <- vapply(seq_len(nrow(methylation_BRCA)), function(i) sum(is.na(methylation_BRCA[i,]))*100/ncol(methylation_BRCA),numeric(1))
max(methylation_miss)
### no missing
sum(is.na(rna))
sum(is.na(mutation))
sum(is.na(copynum_gene))
sum(is.na(copynum_focal))

## create 5-fold cross validation
library(caret)
flds <- createFolds(clinical_BRCA["status",], k = 5, list = TRUE, returnTrain =FALSE)

# setwd("/Users/jieun/Dropbox/Desktop/UNC/Spring2021/COMP790/FinalProject")
save(clinical_BRCA, rna_BRCA,  mutation_BRCA, methylation_BRCA, cnv_gene_BRCA, cnv_focal_BRCA, miRNA_BRCA, flds,
     file = "./data/TCGA-BRCA.rda")
# load("./data/TCGA-BRCA.rda")
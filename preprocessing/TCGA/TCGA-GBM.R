## read in clinical data
clinical <-read.table("C:/Users/wancenmu/Downloads/GBMLGG/Human__TCGA_GBMLGG__MS__Clinical__Clinical__01_28_2016__BI__Clinical__Firehose.tsi",
                      header = T,row.names=1)
clinical <- clinical[-9,]
## read in miRNA data. no missing
miRNA <-read.table("C:/Users/wancenmu/Downloads/GBMLGG/Human__TCGA_GBMLGG__BDGSC__miRNASeq__HS_miR__01_28_2016__BI__Gene__Firehose_RPM_log2.cct",
                      header = T,row.names=1)

## read in methylation all <3%
methylation <-read.table(gzfile("C:/Users/wancenmu/Downloads/GBMLGG/Human__TCGA_GBMLGG__JHU_USC__Methylation__Meth450__01_28_2016__BI__Gene__Firehose_Methylation_Prepocessor.cct.gz"),
                      header = T,row.names=1)

## read in SNP no missing
snp <-read.table(gzfile("C:/Users/wancenmu/Downloads/GBMLGG/Human__TCGA_GBMLGG__BI__SCNA__SNP_6.0__01_28_2016__BI__Gene__Firehose_GISTIC2.cct.gz"),
                         header = T,row.names=1)

## read in RNA no missing
rna <-read.table(gzfile("C:/Users/wancenmu/Downloads/GBMLGG/Human__TCGA_GBMLGG__UNC__RNAseq__HiSeq_RNA__01_28_2016__BI__Gene__Firehose_RSEM_log2.cct.gz"),
                 header = T,row.names=1)

## read in mutation no missing
mutation <-read.table(gzfile("C:/Users/wancenmu/Downloads/GBMLGG/Human__TCGA_GBMLGG__WUSM__Mutation__GAIIx__01_28_2016__BI__Gene__Firehose_MutSig2CV.cbt.gz"),
                 header = T,row.names=1)

sample <- Reduce(intersect, list(colnames(mutation),colnames(rna),colnames(snp),
                                 colnames(methylation),colnames(miRNA),colnames(clinical)))

## filter out unique samples
clinical_GBM <- clinical[,sample]
mutation_GBM <- mutation[,sample]
rna_GBM <- rna[,sample]
snp_GBM <- snp[,sample]
methylation_GBM <- methylation[,sample]
miRNA_GBM <- miRNA[,sample]

## check missingness
### <7%
clinical_miss<-vapply(seq_len(nrow(clinical_GBM)), function(i) sum(is.na(clinical_GBM[i,]))*100/ncol(clinical_GBM),numeric(1))
methylation_miss<-vapply(seq_len(nrow(methylation_GBM)), function(i) sum(is.na(methylation_GBM[i,]))*100/ncol(methylation_GBM),numeric(1))
hist(methylation_miss)
### no missing
sum(is.na(rna))
sum(is.na(snp))
sum(is.na(mutation))

## create 5-fold cross validation
library(caret)
flds <- createFolds(clinical_GBM["status",], k = 5, list = TRUE, returnTrain =FALSE)

save(clinical_GBM,mutation_GBM,rna_GBM,snp_GBM,methylation_GBM,miRNA_GBM,flds,file = "./data/TCGA-GBM.rda")
# load("./data/TCGA-GBM.rda")

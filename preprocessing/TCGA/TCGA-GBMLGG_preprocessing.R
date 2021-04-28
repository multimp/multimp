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
CNV <-read.table(gzfile("C:/Users/wancenmu/Downloads/GBMLGG/Human__TCGA_GBMLGG__BI__SCNA__SNP_6.0__01_28_2016__BI__Gene__Firehose_GISTIC2.cct.gz"),
                         header = T,row.names=1)

## read in RNA no missing
rna <-read.table(gzfile("C:/Users/wancenmu/Downloads/GBMLGG/Human__TCGA_GBMLGG__UNC__RNAseq__HiSeq_RNA__01_28_2016__BI__Gene__Firehose_RSEM_log2.cct.gz"),
                 header = T,row.names=1)

## read in mutation no missing
mutation <-read.table(gzfile("C:/Users/wancenmu/Downloads/GBMLGG/Human__TCGA_GBMLGG__WUSM__Mutation__GAIIx__01_28_2016__BI__Gene__Firehose_MutSig2CV.cbt.gz"),
                 header = T,row.names=1)

## read in SCNV focal no missing
CNV_focal <-read.table("C:/Users/wancenmu/Downloads/GBMLGG/Human__TCGA_GBMLGG__BI__SCNA__SNP_6.0__01_28_2016__BI__Focal__Firehose_GISTIC2.cct",
                      header = T,row.names=1)

sample <- Reduce(intersect, list(colnames(mutation),colnames(rna),colnames(CNV),colnames(CNV_focal),
                                 colnames(methylation),colnames(miRNA),colnames(clinical)))

## filter out unique samples
clinical_GBM <- clinical[,sample]
mutation_GBM <- mutation[,sample]
rna_GBM <- rna[,sample]
CNV_GBM <- CNV[,sample]
CNV_focal_GBM <- CNV_focal[,sample]
methylation_GBM <- methylation[,sample]
miRNA_GBM <- miRNA[,sample]

## QC on RNA
rna_check<- rowSums(rna_GBM)
hist(rna_check,breaks = 100)
keep<-which(rna_check>ncol(rna_GBM))
rna_GBM <- rna_GBM[keep,]

## QC on mutation
mutation_check <- rowSums(mutation_GBM)
hist(mutation_check,breaks = 400,xlim = c(0,40))
keep_mutate <- which(mutation_check>0)
mutation_GBM <- mutation_GBM[keep_mutate,]

## check missingness
### <7%
clinical_miss<-vapply(seq_len(nrow(clinical_GBM)), function(i) sum(is.na(clinical_GBM[i,]))*100/ncol(clinical_GBM),numeric(1))

methylation_miss<-vapply(seq_len(ncol(methylation_GBM)), function(i) sum(is.na(methylation_GBM[i,]))*100/n(methylation_GBM),numeric(1))
hist(methylation_miss)

methylation_miss<-vapply(seq_len(nrow(methylation_GBM)), function(i) sum(is.na(methylation_GBM[i,])),numeric(1))
hist(methylation_miss)
length(which(methylation_miss==0))
### no missing
sum(is.na(rna))
sum(is.na(CNV))
sum(is.na(mutation))

## complete case for all views
index <- complete.cases(t(clinical_GBM))
clinical_GBM2 <- t(clinical_GBM[,index])
mutation_GBM2 <- t(mutation_GBM[complete.cases(mutation_GBM),index])
rna_GBM2 <- t(rna_GBM[complete.cases(rna_GBM),index])
CNV_GBM2 <- t(CNV_GBM[complete.cases(CNV_GBM),index])
CNV_focal_GBM2 <- t(CNV_focal_GBM[complete.cases(CNV_GBM),index])
methylation_GBM2 <- t(methylation_GBM[complete.cases(CNV_GBM),index])
miRNA_GBM <- miRNA[,sample]

## create 5-fold cross validation
library(caret)
clinical_GBM["status",] <-ifelse(is.na(clinical_GBM["status",]),3,clinical_GBM["status",])
flds <- createFolds(clinical_GBM["status",], k = 5, list = TRUE, returnTrain =FALSE)

save(clinical_GBM,mutation_GBM,rna_GBM,CNV_GBM,CNV_focal_GBM,methylation_GBM,miRNA_GBM,flds,file = "./data/TCGA-GBM.rda")
# load("./data/TCGA-GBM.rda")

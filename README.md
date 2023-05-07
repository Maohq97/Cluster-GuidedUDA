# Cluster-Guided UDA

This repository contains the programs and extra experimental results of the paper "[Cluster-Guided Unsupervised Domain Adaptation for Deep Speaker Embedding](https://arxiv.org/abs/2303.15944)".

## Training Framework

The proposed UDA framework mainly involves the following steps:

Step 1: Pre-train a model by jointing the speaker classification loss in the source domain and the contrastive loss in the target domain. (1-Pre_training)

Step 2: Fine-tune the pre-trained model with the additional contrastive center loss. (2-Fine_tuning)

Step 3: Conduct supervised training using labeled source domain data and pseudo-labeled target domain data. (3-Supervised_training)

Each step can be implemented by the given programs. The backbone used in the paper is [ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN).

## VoxCeleb2 - CnCeleb1

Source domain data: [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) (English, 5994 speakers)

Target domain data: [CN-Celeb1](http://www.openslr.org/82/) (Chinese, 797 speakers)

Here is a brief introduction of the experimental results for this part. More details can be seen in the [original paper](https://arxiv.org/abs/2303.15944).

### Baseline

$$
\text{Table 1} \space \space \text{Results on CN-Celeb1 without domain adaptation} \\
\begin{array}
{ccc}
\hline
	\text{Training data} & \text{Training Mode} & \text{EER(\%)} \\
\hline
	\text{VoxCeleb2} & \text{Supervised} & 13.40 \\
	\text{CN-Celeb1} & \text{Supervised} & 11.04 \\
	\text{CN-Celeb1} & \text{CSL}        & 13.36 \\
\hline	
\end{array}
$$

Due to the domain mismatch, the result of supervised training on VoxCeleb2 is unsatisfactory, which is considered the baseline result.   

### The Effectiveness of the Fine-tuned Model for Clustering

We tested the EER performance of different models on the CN-Celeb1 evaluation set. Besides, we used the obtained models to extract embeddings of all the CN-Celeb1 training data, followed by  $k$-means clustering with $K=800$, which means the total number of clusters.
$$
\text{Table 2} \space \space \text{EER and clustering performance of different models} \\
\begin{array}
{cccc}
\hline
	\text{Model} & \text{EER(\%)} & \text{Purity} & \text{NMI}\\ 
\hline
	\text{Pre-trained} & 10.30 & 0.786 & 0.875\\
	\text{Fine-tuned}  & 9.48  & 0.829 & 0.898\\
\hline	
\end{array}
$$
We further analyzed the variations of EER and NMI during the fine-tuning process. We estimated the total number of clusters as 300, 480, and 700, respectively, and conducted several independent fine-tuning experiments.

<img src=".\imgs\EER-CN1.png" alt="EER-CN1" style="zoom:5%;" />

<img src=".\imgs\NMI-CN1.png" alt="NMI-CN1" style="zoom:5%;" />

The fine-tuning process can improve both recognition performance (EER) and clustering performance (NMI), and such improvements are less affected by the misestimation of the number of clusters.

### Supervised results on the combined dataset

To perform domain adaptation, the labeled source domain data are combined with the pseudo-labeled target domain data, and the total number of classes is their sum. Finally, a new speaker embedding network is trained with a classification layer and cross-entropy loss on the combined dataset.

We varied the number of clusters $K$ to explore its effect on the proposed method. The results are shown in the table below, where "CN1" means the CN-Celeb1 training data with true labels and "CN1.U" means these training data with pseudo labels.
$$
\text{Table 3} \space \space \text{Supervised results on the combined dataset} \\
\begin{array}
{ccc}
\hline
	\text{Training data} & \text{Training Mode} & \text{EER(\%)} \\
\hline
	\text{Vox2+CN1}           & \text{Supervised} & 8.05 \\
	\text{Vox2+CN1.U(K=400)}  & \text{Supervised} & 8.38 \\
	\text{Vox2+CN1.U(K=800)}  & \text{Supervised} & 8.10 \\
	\text{Vox2+CN1.U(K=1600)} & \text{Supervised} & 8.99 \\
\hline	
\end{array}
$$
The supervised training on the combined dataset brings significant performance improvement even when the number of clusters deviates from the true one. Besides, the result ($K$=800) is even close to the upper bound performance when the actual labels of the target domain data are available.

## VoxCeleb2 - aidatatang_200zh

Source domain data: [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) (English, 5994 speakers)

Target domain data: [aidatatang_200zh](http://www.openslr.org/62/) (Chinese, 480 speakers)

To demonstrate that the proposed method works not only on Cn-Celeb1 but also on other datasets, we have conducted experiments on aidatatang_200zh, a Chinese Mandarin speech corpus. Due to the page limitation of the submitted journal, we show experimental results for this part here, rather than in the original paper.

The aidatatang_200zh dataset contains 200 hours of acoustic data from 600 speakers. In our experiment, we used 480 speakers for training and 120 speakers for evaluation. For each speaker in the evaluation set, the utterance with the longest duration is selected as the enrollment utterance, and the remaining  parts are used as the test utterances. The trial list contains 5,762,877 trial pairs in total.

### Baseline

$$
\text{Table 4} \space \space \text{Results on aidatatang\_200zh without domain adaptation} \\
\begin{array}
{ccc}
\hline
	\text{Training Data} & \text{Training Mode} & \text{EER(\%)} \\
\hline
	\text{VoxCeleb2}         & \text{Supervised} & 10.30 \\
	\text{aidatatang\_200zh} & \text{CSL}        & 4.47  \\
\hline	
\end{array}
$$

### The Effectiveness of the Fine-tuned Model for Clustering

We tested the EER performance of different models on the aidatatang_200zh evaluation set.  The obtained models are used to extract embeddings of all the aidatatang_200zh training data, followed by $k$-means clustering with $K=480$.  
$$
\text{Table 5} \space \space \text{EER and clustering performance of different models} \\
\begin{array}
{cccc}
\hline
	\text{Model} & \text{EER(\%)} & \text{Purity} & \text{NMI}\\ 
\hline
	\text{Pre-trained} & 4.28 & 0.770 & 0.951\\
	\text{Fine-tuned}  & 4.13 & 0.809 & 0.958\\
\hline	
\end{array}
$$
Similarly, we analyzed the variations of EER and NMI during the fine-tuning process.  The total number of clusters was estimated to be 300, 480, and 700, respectively.

<img src=".\imgs\EER-adt.png" alt="EER-adt" style="zoom:7%;" />

<img src=".\imgs\NMI-adt.png" alt="NMI-adt" style="zoom:7%;" />

Fine-tuning can effectively improve the model recognition performance as well as the clustering quality, and it also brings a stable improvement even when the total number of clusters deviates from the true value.

### Supervised results on the combined dataset

The supervised results on the combined dataset are listed below, where "aidatatang" means the aidatatang_200zh training data with true labels and "aidatatang.U" means these data with pseudo labels.
$$
\text{Table 6} \space \space \text{Supervised results on the combined dataset} \\
\begin{array}
{ccc}
\hline
	\text{Training Data} & \text{Training Mode} & \text{EER(\%)} \\
\hline
	\text{Vox2+aidatatang}           & \text{Supervised} & 3.15 \\
	\text{Vox2+aidatatang.U(K=300)}  & \text{Supervised} & 3.18 \\
	\text{Vox2+aidatatang.U(K=480)}  & \text{Supervised} & 3.29 \\
	\text{Vox2+aidatatang.U(K=700)}  & \text{Supervised} & 3.49 \\
\hline	
\end{array}
$$

The experimental conclusions are consistent with those on CN-Celeb1, which further verify that the proposed UDA framework can maximally leverage the labeled source domain data and the unlabeled target domain data, improving system performance on the mismatched target domain. 


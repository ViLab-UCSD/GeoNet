import torch.nn as nn
import torch.nn.functional as F
import torch

def cdist(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    res = res.clamp_min_(1e-30).sqrt_()
    return res

class MSCLoss(nn.Module):
    def __init__(self, n_neighb=5, distance="cosine", temperature=0.07):
        super().__init__()
        self.ranking_k = 4
        self.eps = 1e-9
        self.similarity_func = distance  # euclidean dist, cosine
        self.top_n_sim =n_neighb
        self.conf_ind = None
        self.tau = temperature

    def __get_sim_matrix(self, out_src, out_tar):
        matrix = None
        if (self.similarity_func == 'euclidean'): ## Inverse Euclidean Distance
            matrix = cdist(out_src, out_tar)
            matrix = matrix + 1.0
            matrix = 1.0/matrix

        elif (self.similarity_func == 'gaussian'): ## exponential Gaussian Distance
            matrix = cdist(out_src, out_tar)
            matrix = torch.exp(-1*matrix)

        elif (self.similarity_func == 'cosine'): ## Cosine Similarity
            out_src = F.normalize(out_src, dim=1, p=2)
            out_tar = F.normalize(out_tar, dim=1, p=2)
            matrix = torch.matmul(out_src, out_tar.T)

        else:
            raise NotImplementedError

        return matrix

    #func to assign target labels by KNN
    def assign_labels_KNN(self, sim_matrix, src_labels):

        ind = torch.sort(sim_matrix, descending=True, dim=0).indices
        k_orderedNeighbors = src_labels[ind[:self.top_n_sim]]
        assigned_target_labels = torch.mode(k_orderedNeighbors, dim=0).values

        return assigned_target_labels, ind
    
    def calc_loss(self, confident_sim_matrix, src_labels, confident_tgt_labels):
        n_src = src_labels.shape[0]
        n_tgt = confident_tgt_labels.shape[0]
        
        vr_src = src_labels.unsqueeze(-1).repeat(1, n_tgt)
        hr_tgt = confident_tgt_labels.unsqueeze(-2).repeat(n_src, 1)
        
        mask_sim = (vr_src == hr_tgt).float()

        expScores = torch.softmax(confident_sim_matrix/self.tau, dim=0)
        contrastiveMatrix = (expScores * mask_sim).sum(0) / (expScores.sum(0))
        MSC_loss = -1 * torch.mean(torch.log(contrastiveMatrix + 1e-6))
        
        return MSC_loss

    def forward(self, source_features, source_labels, target_features):

        n_tgt = len(target_features)
        top_ranked_n = n_tgt*2//3

        sim_matrix = self.__get_sim_matrix(source_features, target_features)
        flat_src_labels = source_labels.squeeze()

        assigned_tgt_labels, sorted_indices  = self.assign_labels_KNN(sim_matrix, source_labels)
        self.all_assigned = assigned_tgt_labels

        ranking_score_list = []

        for i in range(0, n_tgt): #nln: nearest like neighbours, nun: nearest unlike neighbours
            nln_mask = (flat_src_labels == assigned_tgt_labels[i]).float()
            sorted_nln_mask = nln_mask[sorted_indices[:,i]].bool()
            nln_sim_r  = sim_matrix[:,i][sorted_indices[:,i][sorted_nln_mask]][:self.ranking_k]

            nun_mask = ~(flat_src_labels == assigned_tgt_labels[i])
            nun_mask = nun_mask.float()
            sorted_nun_mask = nun_mask[sorted_indices[:,i]].bool()
            nun_sim_r  = sim_matrix[:,i][sorted_indices[:,i][sorted_nun_mask]][:self.ranking_k]

            pred_conf_score = (1.0*torch.sum(nln_sim_r)/torch.sum(nun_sim_r)).detach() #sim ratio : confidence score
            ranking_score_list.append(pred_conf_score)

        top_n_tgt_ind = torch.topk(torch.tensor(ranking_score_list), top_ranked_n)[1]
        confident_sim_matrix = sim_matrix[:, top_n_tgt_ind]
        confident_tgt_labels = assigned_tgt_labels[top_n_tgt_ind] #filtered tgt labels
        loss_targetAnch = self.calc_loss(confident_sim_matrix, source_labels, confident_tgt_labels)

        return loss_targetAnch
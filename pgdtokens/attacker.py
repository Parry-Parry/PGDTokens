import torch 
from torch import nn

from pgdtokens.pgd import PGD

class Attacker:
    def __init__(self):
        pass
    def rank_attack_loss(self, query_rep : torch.Tensor, pos_rep : torch.Tensor, neg_rep : torch.Tensor):
        '''

        :param query_rep: shape = [1, dim]
        :param pos_rep: shape = [1, dim]
        :param neg_rep: shape = [B, dim]
        :return:
        '''

        margin = 1.0

        reduction = 'sum'
        loss_fct = nn.MarginRankingLoss(margin=margin, reduction=reduction)

        num_neg = neg_rep.shape[0]
        
        pos_s = torch.mm(query_rep, pos_rep.transpose(0, 1))
        neg_s = torch.mm(query_rep.repeat(num_neg, 1), neg_rep.transpose(0, 1))

        pos_s = pos_s.repeat(num_neg, 1)

        labels = torch.ones_like(pos_s)
        computed_loss = loss_fct(pos_s, neg_s, labels)
        return computed_loss

    def get_model_gradient(self, model, query, docs):
        model.train()
        model.zero_grad()

        query, docs = query.to(model.device), docs.to(model.device)

        query_rep = model(**query)[0, :]
        pos_rep = model(**docs[0])[0, :]
        neg_rep = model(**docs[1:])[:, 0, :]

        loss = self.rank_attack_loss(query_rep, pos_rep, neg_rep)
        if loss is not None:
            loss.backward()


    def attack(self, 
               model, 
               query,
               docs,  
               attack_word_idx, 
               name,
               eps=0.009, 
               max_iter=3):

        model.train()

        pgd_attacker = PGD(model, attack_word_idx)

        alpha = eps / max(1, max_iter//2)
        for t in range(max_iter):

            self.get_model_gradient(model, query, docs)

            pgd_attacker.attack(is_first_attack=(t == 0), epsilon=eps,
                                alpha=alpha, emb_name=name)

            model.zero_grad()

    def random_attack(self, model, name):

        emb_name = name
        # change the emb_name to the embedding parameter name of your model
        for name, param in model.named_parameters():
            if emb_name in name:
                param_avg = torch.mean(param)
                r_raodong = ((torch.rand(param.shape)-0.5) * 2).to(model.device) * param_avg
                param.data.add_(r_raodong)

    def attack_with_momentum(self, 
                             model, 
                             batch_list, 
                             attack_doc_input, 
                             attack_word_idx, 
                             name,
                             eps=0.009, 
                             max_iter=3,
                             momentum=0):
        model.train()
        pgd_attacker = PGD(model, attack_word_idx, momentum=momentum)
        for batch in batch_list:

            batch = {k: torch.cat((attack_doc_input[k].unsqueeze(dim=0), v), dim=0)
                     for k, v in batch.items()}
            batch = {k: v.to(model.device) for k, v in batch.items()}

            alpha = eps / max(1, max_iter//2)
            for t in range(max_iter):

                outputs = model(**batch)
                attack_loss = self.rank_attack_loss(outputs[0], outputs[1:])

                if attack_loss is not None:
                    attack_loss.backward()

                pgd_attacker.attack_with_momentum(is_first_attack=(t == 0),
                                                  epsilon=eps, alpha=alpha,
                                                  emb_name=name)

                model.zero_grad()
import torch 

class WordRecover:
    def __init__(self, embed_name):
        self.embed_name = embed_name
        pass

    def get_word_embedding(self, model):
        emb_name = self.embed_name
        for name, param in model.named_parameters():
            if emb_name in name:
               embedding_matrix = param
               return embedding_matrix

    def get_max_sim_word(self, word_tensors, ori_word_embedding_matrix):
        cossim_maxtrix = self.cos_similar(word_tensors, ori_word_embedding_matrix)
        max_sim, max_idx = torch.max(cossim_maxtrix, dim=-1)

        return max_sim, max_idx

    def cos_similar(self, p:torch.Tensor, q:torch.Tensor):
        sim_maxtrix = p.matmul(q.transpose(-2, -1))
        a = torch.norm(p, p=2, dim=-1)
        b = torch.norm(q, p=2, dim=-1)
        sim_maxtrix /= a.unsqueeze(-1)
        sim_maxtrix /= b.unsqueeze(-2)
        return sim_maxtrix

    def get_sim_word_matrix(self, simwords_list, ori_word_embedding_matrix):
        candidate_word_matrix_list = {}
        candidate_word_no_dict = {}
        for ori_word in simwords_list:
            idx_list = []
            simwords = simwords_list[ori_word]
            for simword in simwords:
                simword_idx = self.tokenizer.encode(simword)
                if len(simword_idx) > 1:
                    continue
                idx_list.append(simword_idx[0])
            simword_matrix = ori_word_embedding_matrix[idx_list, :]
            candidate_word_matrix_list[ori_word] = simword_matrix
            candidate_word_no_dict[ori_word] = idx_list
        return candidate_word_matrix_list, candidate_word_no_dict

    def get_sim_word_semantic(self, candidate_word_matrix_list, now_word_embedding_matrix):
        max_sim_list = {}
        max_id_list = {}

        for ori_word in candidate_word_matrix_list:
            ori_word_idx = self.tokenizer.encode(ori_word)
            ori_word_tensor = now_word_embedding_matrix[ori_word_idx, :]
            if len(candidate_word_matrix_list[ori_word]) == 0:
                continue
            max_sim, max_idx = self.get_max_sim_word(ori_word_tensor, candidate_word_matrix_list[ori_word])
            max_sim_list[ori_word] = max_sim
            max_id_list[ori_word] = max_idx
        return max_sim_list, max_id_list

    def recover_document_semantic(self, old_doc, ori_wem, now_wem, simwords_list):

        max_word_no_dict = {}
        candidate_word_matrix_dict, candidate_word_no_dict = self.get_sim_word_matrix(simwords_list, ori_wem)
        max_sim_dict, max_id_dict = self.get_sim_word_semantic(candidate_word_matrix_dict, now_wem)
        for word in max_id_dict:
            max_word_no_dict[word] = [candidate_word_no_dict[word][max_id_dict[word][0]]]

        new_doc_list = []
        for token in old_doc.split(' '):
            if token not in max_word_no_dict:
                new_doc_list.append(token)
            else:
                # print(max_word_no_dict[token])
                new_token = self.tokenizer.decode(max_word_no_dict[token])
                # print(new_token)
                new_doc_list.append(new_token)
        new_doc = ' '.join(new_doc_list)
        print(new_doc)
        return new_doc

    def get_highest_gradient_words(self, model, attack_num=50, attack_word_idx = []):
        embed_name = self.embed_name
        word_embedding = self.get_word_embedding(model, embed_name)
        gradient_matrix = word_embedding.grad[attack_word_idx]

        row_norm = torch.norm(gradient_matrix, p=2, dim=-1)

        attack_num = min(attack_num, row_norm.shape[0])

        topk_norm, topk_idx = torch.topk(row_norm, k=attack_num, dim=-1)
        print(topk_norm, topk_idx)
        model.zero_grad()

        word_topk_list = []
        for idx in topk_idx:
            idx = idx.item()
            true_idx = attack_word_idx[idx]
            word_topk_list.append(true_idx)

        return topk_norm, word_topk_list

class BERTWordRecover(WordRecover):
    def __init__(self, embed_name, bert_tokenizer):
        super(BERTWordRecover, self).__init__(embed_name)
        self.tokenizer = bert_tokenizer
        self.word2idx = self.tokenizer.get_vocab()
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.max_query_length = 30
        self.max_doc_length = 256

    def get_word_embedding(self, model):
        emb_name = self.embed_name
        for name, param in model.named_parameters():
            if emb_name in name:
               embedding_matrix = param
               return embedding_matrix

    def get_max_sim_word(self, word_tensors, ori_word_embedding_matrix):

        if len(word_tensors.shape) == 1:

            word_tensors = word_tensors.unsqueeze(0)
        cossim_maxtrix = self.cos_similar(word_tensors, ori_word_embedding_matrix)

        max_sim, max_idx = torch.max(cossim_maxtrix, dim=-1)

        return max_sim, max_idx

    def cos_similar(self, p:torch.Tensor, q:torch.Tensor):
        sim_maxtrix = p.matmul(q.transpose(-2, -1))
        a = torch.norm(p, p=2, dim=-1)
        b = torch.norm(q, p=2, dim=-1)
        sim_maxtrix /= a.unsqueeze(-1)
        sim_maxtrix /= b.unsqueeze(-2)
        return sim_maxtrix

    def get_sim_word_matrix(self, simword_dict, ori_word_embedding_matrix):

        candidate_word_matrix_dict = {}
        candidate_word_index_dict = {}
        zhanwei_token_id = self.tokenizer.cls_token_id
        for ori_word in simword_dict:
            idx_list = []
            simwords = simword_dict[ori_word]
            multi_idx_list = []
            multi_idx_list_index_list = []
            index = 0
            for simword in simwords:

                simword_idx = self.tokenizer.encode(simword, add_special_tokens=False)

                if len(simword_idx) > 1:
                    multi_idx_list.append(simword_idx)
                    multi_idx_list_index_list.append(index)
                    idx_list.append(zhanwei_token_id)
                else:
                    idx_list.append(simword_idx[0])
                index += 1

            simword_matrix = ori_word_embedding_matrix[idx_list, :]
            assert torch.equal(simword_matrix[multi_idx_list_index_list],
                               ori_word_embedding_matrix[[zhanwei_token_id] * len(multi_idx_list_index_list)]) == True
            m_index = 0
            for multi_idx in multi_idx_list:
                multi_sim_vectors = ori_word_embedding_matrix[multi_idx, :]
                sim_vector = torch.mean(multi_sim_vectors, dim=0, keepdim=False)
                simword_matrix[multi_idx_list_index_list[m_index]] = sim_vector
                idx_list[multi_idx_list_index_list[m_index]] = multi_idx
                m_index += 1

            candidate_word_matrix_dict[ori_word] = simword_matrix

            candidate_word_index_dict[ori_word] = idx_list
        return candidate_word_matrix_dict, candidate_word_index_dict

    def get_sim_word_semantic(self, candidate_word_matrix_dict, now_word_embedding_matrix):

        max_sim_dict = {}
        max_id_dict = {}

        for ori_word in candidate_word_matrix_dict:

            if ori_word not in self.word2idx:
                ori_word_ids = self.tokenizer.encode(ori_word, add_special_tokens=False)
                ori_word_tensor = now_word_embedding_matrix[ori_word_ids, :]
                ori_word_tensor = torch.mean(ori_word_tensor, dim=0, keepdim=False)
            else:
                ori_word_id = self.word2idx[ori_word]
                ori_word_tensor = now_word_embedding_matrix[ori_word_id, :]
            if len(candidate_word_matrix_dict[ori_word]) == 0:
                continue
            max_sim, max_idx = self.get_max_sim_word(ori_word_tensor, candidate_word_matrix_dict[ori_word])
            max_sim_dict[ori_word] = max_sim
            max_id_dict[ori_word] = max_idx
        return max_sim_dict, max_id_dict

    def recover_document_semantic(self, old_doc_token_list, ori_wem, now_wem, simword_dict):

        max_word_index_dict = {}

        candidate_word_matrix_dict, candidate_word_index_dict = self.get_sim_word_matrix(simword_dict, ori_wem)
        max_sim_dict, max_id_dict = self.get_sim_word_semantic(candidate_word_matrix_dict, now_wem)

        for word in max_id_dict:
            max_word_index_dict[word] = [candidate_word_index_dict[word][max_id_dict[word][0]]]

        new_doc_token_id_list = []
        old_doc = self.tokenizer.decode(old_doc_token_list)
        old_doc_token_words = old_doc.split(' ')
        for old_doc_word in old_doc_token_words:
            if old_doc_word not in max_word_index_dict:
                new_doc_token_id_list.append(old_doc_word)
            else:
                # print(max_word_no_dict[token])
                new_token_id = max_word_index_dict[old_doc_word]
                # print(new_token)
                new_doc_token_id_list.append(new_token_id[0])

        return new_doc_token_id_list

    def pack_tensor_2D(self, lstlst, default, dtype, length=None):
        batch_size = len(lstlst)
        length = length if length is not None else max(len(l) for l in lstlst)
        tensor = default * torch.ones((batch_size, length), dtype=dtype)
        for i, l in enumerate(lstlst):
            # print(i, l)
            tensor[i, :len(l)] = torch.tensor(l, dtype=dtype)
        return tensor

    def eval_model(self, model, doc_input_ids, query_input_ids):

        query_input_ids = query_input_ids[:self.max_query_length]
        query_input_ids = [self.cls_id] + query_input_ids 
        doc_input_ids = [self.cls_id] + doc_input_ids[:self.max_doc_length]
        
        model.eval()

        with torch.no_grad():
            query_input_ids, doc_input_ids = query_input_ids.to(model.device), doc_input_ids.to(model.device)
            s = torch.mm(query_input_ids.repeat(doc_input_ids.shape[0], 1), doc_input_ids.transpose(0, 1))
        return s.detach().cpu().numpy()

    def find_ori_word_index(self, ori_doc_token_ids_list, specific_token_ids_list):
        # O(n）
        finded_start_index_list = []
        for i in range(len(ori_doc_token_ids_list)):
            if ori_doc_token_ids_list[i] == specific_token_ids_list[0]:
                ori_sub_tokens = ori_doc_token_ids_list[i: i + len(specific_token_ids_list)]
                if ori_sub_tokens == specific_token_ids_list:
                    finded_start_index_list.append(i)
        return finded_start_index_list

    def replace_one_word_token_ids(self, doc_token_id_list, word_token_ids, new_token_ids, finded_start_index_list):

        res_token_id_list = []

        index = finded_start_index_list[0]
        finded_start_index_list.pop(0)

        assert doc_token_id_list[index] == word_token_ids[0]

        res_token_id_list.extend(doc_token_id_list[: index])
        res_token_id_list.extend(new_token_ids)
        last_index = index + len(word_token_ids)

        assert last_index <= len(doc_token_id_list)
        res_token_id_list.extend(doc_token_id_list[last_index:])

        for i in range(len(finded_start_index_list)):
            finded_start_index_list[i] = finded_start_index_list[i] + len(new_token_ids) - len(word_token_ids)

        return res_token_id_list, finded_start_index_list

    def get_rank_pos(self, ori_qds, score, attack_doc_id, qid):
        ranked_list = ori_qds[qid]
        ranked_list[attack_doc_id] = score
        sorted_ranked_list = sorted(ranked_list.items(), key=lambda item: item[1], reverse=True)
        index = 0
        for did_score in sorted_ranked_list:
            index += 1
            did = did_score[0]
            if did == attack_doc_id:
                break
        return index

    def recover_document_greedy_rank_pos(self, old_doc_token_list, ori_wem,
                                         now_wem, simword_dict, ori_score,
                                         model, query_input_id, args, subword_dict,
                                         ori_qds, attack_doc_id, qid):

        max_word_index_dict = {}

        candidate_word_matrix_dict, candidate_word_index_dict = self.get_sim_word_matrix(simword_dict, ori_wem)
        max_sim_dict, max_id_dict = self.get_sim_word_semantic(candidate_word_matrix_dict, now_wem)

        for word in max_id_dict:
            max_word_index_dict[word] = [candidate_word_index_dict[word][max_id_dict[word][0]]]

        m = args.max_attack_word_number
        current_attacked_num = 0
        import copy
        doc_token_id_list = copy.deepcopy(old_doc_token_list)
        best_score = ori_score

        ori_pos = self.get_rank_pos(ori_qds, ori_score, attack_doc_id, qid)
        best_pos = ori_pos

        for word in max_id_dict:
            new_doc_token_id_list = copy.deepcopy(doc_token_id_list)

            if word in subword_dict:
                word_token_ids = subword_dict[word]
            else:
                if word not in self.word2idx:
                    print(word)
                word_token_ids = [self.word2idx[word]]
            new_token_ids = max_word_index_dict[word]
            if isinstance(new_token_ids[0], list):
                new_token_ids = new_token_ids[0]
            finded_start_index_list = self.find_ori_word_index(new_doc_token_id_list,
                                                               word_token_ids)
            while finded_start_index_list:
                new_doc_token_id_list, finded_start_index_list = self.replace_one_word_token_ids(new_doc_token_id_list,
                                                                                   word_token_ids,
                                                                                   new_token_ids,
                                                                                   finded_start_index_list)
                # inference
                score = self.eval_model(model, new_doc_token_id_list, query_input_id, args)[0][0]

                rank_pos = self.get_rank_pos(ori_qds, score, attack_doc_id, qid)
                if rank_pos < best_pos:
                    best_pos = rank_pos
                    best_score = score
                    doc_token_id_list = new_doc_token_id_list
                    current_attacked_num += 1
                if current_attacked_num == m:
                    break
            if current_attacked_num == m:
                break

        return doc_token_id_list, best_score

    def get_highest_gradient_words(self, model, attack_num=50, attack_word_idx = []):

        word_embedding = self.get_word_embedding(model)
        gradient_matrix = word_embedding.grad[attack_word_idx]

        row_norm = torch.norm(gradient_matrix, p=2, dim=-1)

        attack_num = min(attack_num, row_norm.shape[0])

        topk_norm, topk_idx = torch.topk(row_norm, k=attack_num, dim=-1)

        model.zero_grad()

        word_topk_list = []
        for idx in topk_idx:
            idx = idx.item()
            true_idx = attack_word_idx[idx]
            word_topk_list.append(true_idx)

        return topk_norm, word_topk_list

    def recover_doc(self, doc_id, attacked_doc_token_ids, collection, max_doc_length):
        if self.sep_id in attacked_doc_token_ids:
            sep_index = attacked_doc_token_ids.index(self.sep_id)
            attacked_doc_token_ids = attacked_doc_token_ids[:sep_index]

        attacked_doc_token_ids = attacked_doc_token_ids + collection[int(doc_id)][max_doc_length:]
        attacked_doc = self.tokenizer.decode(attacked_doc_token_ids)
        return attacked_doc
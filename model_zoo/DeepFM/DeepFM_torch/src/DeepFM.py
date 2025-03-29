# =========================================================================
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
import json
import numpy as np
import pandas as pd
import torch
import random
from torch import nn
import torch.nn.functional as F
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, FactorizationMachine,MLP_Split_Block


class DeepFM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DeepFM", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 hidden_units=[64, 64, 64], 
                 hidden_activations="ReLU", 
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 bn_dropout = 0,
                 alpha_e = 1,
                 alpha_i = 1,
                 alpha_i_aug = 1,
                 alpha_e_aug = 1,
                 gamma = 2,
                 **kwargs):
        super(DeepFM, self).__init__(feature_map, 
                                     model_id=model_id, 
                                     gpu=gpu, 
                                     embedding_regularizer=embedding_regularizer, 
                                     net_regularizer=net_regularizer,
                                     **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        # self.embedding_layer2 = FeatureEmbedding(feature_map, embedding_dim)
        self.fm = FactorizationMachine(feature_map)
        # self.fm1 = FactorizationMachine(feature_map)
        self.mlp = MLP_Block(input_dim=feature_map.sum_emb_out_dim(),
                             output_dim=1, 
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm)


        # self.mlp1 = MLP_Block(input_dim=feature_map.sum_emb_out_dim(),
        #                      output_dim=1, 
        #                      hidden_units=hidden_units,
        #                      hidden_activations=hidden_activations,
        #                      output_activation=None, 
        #                      dropout_rates=net_dropout, 
        #                      batch_norm=batch_norm)
      

        self.fm_bn = nn.BatchNorm1d(1)
        self.mlp_bn = nn.BatchNorm1d(1)


        self.bn_dropout = bn_dropout

        self.alpha_e = alpha_e
        self.alpha_i = alpha_i

        self.alpha_i_aug = alpha_i_aug
        self.alpha_e_aug = alpha_e_aug

        # self.feature_vocab = self.read_json("../../../data/Movielens/movielenslatest_x1_cd32d937/feature_vocab.json")
        # self.item_vocab_len = len(self.feature_vocab['item_id']) #23606
        # self.tag_vocab_len = len(self.feature_vocab['tag_id']) #49659
        # self.user_vocab_len = len(self.feature_vocab['user_id']) #16977 (包括了第一个pad和最后的oov)
        # self.item_tag = self.read_json("../../../data/Movielens/MovielensLatest_x1/grouped_data.json")
        # self.item_vocab = self.feature_vocab['item_id']
        # self.tag_vocab = self.feature_vocab['tag_id']
        # self.user_vocab = self.feature_vocab['user_id']
        # self.item_vocab_reversed = {v: k for k, v in self.item_vocab.items()}

        # self.sigma_one =  nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        # self.sigma_two = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

        # self.gamma = gamma

        # # self.focal_loss = Focal_Loss(alpha=0.25, gamma=self.gamma)
        self.focal_loss = Focal_Loss_two(alpha=0.5, gamma=2)
        
        # self.fm_ln = nn.LayerNorm(embedding_dim)
        # self.mlp_ln = nn.LayerNorm(embedding_dim)
        # self.mlp2 = MLP_Block(input_dim=feature_map.sum_emb_out_dim(),
        #                      output_dim=1, 
        #                      hidden_units=hidden_units,
        #                      hidden_activations=hidden_activations,
        #                      output_activation=None, 
        #                      dropout_rates=net_dropout, 
        #                      batch_norm=batch_norm)
        # self.projector1 = nn.Linear(1, embedding_dim) #会增强模型性能不知道为啥
        # self.projector2 = nn.Linear(1, embedding_dim)
        self.act = nn.Tanh()
        # self.projector3 = nn.Linear(1, embedding_dim)
        # self.mlp = MLP_Split_Block(input_dim=feature_map.sum_emb_out_dim(),
        #                                  output_dim=1, 
        #                                  hidden_units=hidden_units,
        #                                  hidden_activations=hidden_activations,
        #                                  output_activation=None,
        #                                  dropout_rates=net_dropout,
        #                                  batch_norm=batch_norm
        #                                 )
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    # def listmle(self, y_pred, y_true):
    #     sorted_indices = torch.argsort(y_true, dim =0,descending=True)
    #     sorted_pred = y_pred[sorted_indices].view(-1,1)

    #     loss = torch.tensor(1.0)
    #     for i in range(sorted_pred.shape[0]):
    #         sum_from_i = torch.exp(sorted_pred[i:]).sum() 
    #         loss = loss * (torch.exp(sorted_pred[i]) / sum_from_i)

    #     loss_log = -torch.log(loss+ 1e-10)
    #     return loss_log
    
    def listmle(self,y_pred, y_true):
        # k = 1024
        # if k is not None:
        #     sublist_indices = torch.randperm(y_true.shape[0])[:k]
        #     # print("sublist_indices:",sublist_indices)
        #     y_pred = y_pred[sublist_indices]
        #     y_true = y_true[sublist_indices]

        sorted_indices = torch.argsort(y_true, dim =0,descending=True)

        sorted_pred = y_pred[sorted_indices].view(-1,1)

        cumsums = sorted_pred.exp().flip(dims=[0]).cumsum(dim=0).flip(dims=[0])

        loss_log = torch.log(cumsums) - sorted_pred

    
        return loss_log.mean()

    def read_json(self, json_file):
  
        # 读取JSON文件
        with open(json_file, 'r') as file:
            data = json.load(file)
        # print(data)
        print("Loading feature_vocab.json...")
        return data

    def data_augmentation(self,inputs):
        raw_data = inputs.cpu().numpy()

        augmented_data = []
        for idx, row in enumerate(raw_data):
            user_id = row[0]
            
            # 如果当前user_id是非最后一个，就取下一个user_id对应的数据进行增广
            if idx != len(raw_data) - 1:
                next_user_id = raw_data[idx + 1][0]
                # 如果下一个user_id与当前user_id相等，表示他们是同一个用户的不同交互记录
                # 则跳过当前行，考虑下一行
                if next_user_id == user_id:
                    continue
                next_item_id = raw_data[idx + 1][1]
                next_tag_id = raw_data[idx + 1][2]
            
            # 如果当前user_id是最后一个，就取第一个user_id对应的数据进行增广
            else:
                next_item_id = raw_data[0][1]
                next_tag_id = raw_data[0][2]

            augmented_data.append([user_id, next_item_id, next_tag_id])

        # Convert the augmented data into a tensor 
        augmented_data_tensor = torch.tensor(augmented_data, device=inputs.device, dtype=inputs.dtype)
    
        # Concatenate the original inputs and the augmented data
        # augmented_inputs = torch.cat((inputs, augmented_data_tensor), dim=0)
        
        return augmented_data_tensor 
    
    def data_augmentation(self,inputs):
        raw_data = inputs.cpu().numpy()

        augmented_data = []
        for idx, row in enumerate(raw_data):
            user_id = row[0]
            
            # 如果当前user_id是非最后一个，就取下一个user_id对应的数据进行增广
            if idx != len(raw_data) - 1:
                next_user_id = raw_data[idx + 1][0]
                # 如果下一个user_id与当前user_id相等，表示他们是同一个用户的不同交互记录
                # 则跳过当前行，考虑下一行
                if next_user_id == user_id:
                    continue
                next_item_id = raw_data[idx + 1][1]
                next_tag_id = raw_data[idx + 1][2]
            
            # 如果当前user_id是最后一个，就取第一个user_id对应的数据进行增广
            else:
                next_item_id = raw_data[0][1]
                next_tag_id = raw_data[0][2]

            augmented_data.append([user_id, next_item_id, next_tag_id])

        # Convert the augmented data into a tensor 
        augmented_data_tensor = torch.tensor(augmented_data, device=inputs.device, dtype=inputs.dtype)
    
        # Concatenate the original inputs and the augmented data
        # augmented_inputs = torch.cat((inputs, augmented_data_tensor), dim=0)
        
        return augmented_data_tensor 

    def gene_data(self):

        user_id = torch.randint(low=1, high=self.user_vocab_len-2, size=(1024,), dtype=torch.float64)
        item_id = torch.randint(low=1, high=self.item_vocab_len-2, size=(1024,), dtype=torch.float64)
        tag_id = torch.randint(low=1, high=self.tag_vocab_len-2, size=(1024,), dtype=torch.float64)

        inputs_aug = torch.cat([user_id.unsqueeze(1), item_id.unsqueeze(1), tag_id.unsqueeze(1)], dim=1)

        return inputs_aug


    def gene_data3(self,inputs):
        user_id = inputs[:,0]
        item_id = inputs[:,1]
        tag_id = inputs[:,2]

        # print(user_id.shape, item_id.shape, tag_id.shape)
        # print(user_id[:10], item_id[:10], tag_id[:10])
        genre_size = 100

        user_id_indice = torch.randint(0,len(user_id),(genre_size,))
        item_id_indice = torch.randint(0,len(item_id),(genre_size,))
        tag_id_indice = torch.randint(0,len(tag_id),(genre_size,))

        # print(user_id_indice[:10],item_id_indice[:10],tag_id_indice[:10])
        user_id = user_id[user_id_indice]
        item_id = item_id[item_id_indice]
        tag_id = tag_id[tag_id_indice]

        # user_id = random.choice(user_id, 1024, replace=False)
        # item_id = random.choice(item_id, 1024, replace=False)
        # tag_id = random.choice(tag_id, 1024, replace=False)



        # print(user_id[:10],item_id[:10],tag_id[:10])
        inputs_aug = torch.cat([user_id.unsqueeze(1), item_id.unsqueeze(1), tag_id.unsqueeze(1)], dim=1)


        return inputs_aug


    def min_max_norm (self,inputs):
        tensor_min = inputs.min(dim=0, keepdim=True)[0]
        tensor_max = inputs.max(dim=0, keepdim=True)[0]

        # 使用最大最小归一化公式进行归一化处理
        normalized_tensor = (inputs- tensor_min) / (tensor_max - tensor_min)

        return normalized_tensor

    def forward(self, inputs):
        """
        Inputs: [X,y]
        """

        # print("inputs",inputs)

        # inputs_aug = self.data_augmentation(inputs) #增广数据的效果和我想象的一致
        # inputs_aug = self.gene_data2()
        # inputs_aug = self.gene_data3(inputs)

        # print("inputs_aug",inputs_aug)
        # print(inputs.shape,inputs_aug.shape)

        X = self.get_inputs(inputs)

        # X_aug = self.get_inputs(inputs_aug)
        # print("X is",X) #[user_id,item_id,tag_id]
    #     X is {'user_id': tensor([  29.,  274., 2903.,  ...,  568.,   46.,  103.], device='cuda:0',
    #    dtype=torch.float64), 'item_id': tensor([ 394.,  401.,  262.,  ..., 5099.,  225.,  989.], device='cuda:0',
    #    dtype=torch.float64), 'tag_id': tensor([3.0000e+00, 2.4190e+03, 3.9084e+04,  ..., 3.8318e+04, 1.5000e+01,
    #     2.0070e+03], device='cuda:0', dtype=torch.float64)}

        # Y = self.get_labels(inputs)
        # print("label is",Y[:10])
        feature_emb = self.embedding_layer(X) 
       
        # with torch.no_grad():
        #     X_aug_emb = self.embedding_layer(X_aug)
        # X_aug_emb = X_aug_emb[torch.randperm(X_aug_emb.shape[0])[:1024],:]


        # print("feature_emb is",feature_emb) #[batch_size, feature_num, embedding_dim]
        # print(X_aug_emb.shape)  #[4081,3,10]这样子
# feature_emb is tensor([[[-0.0018,  0.0005, -0.0090,  ..., -0.0053,  0.0018,  0.0039],
#          [-0.0049, -0.0031,  0.0004,  ..., -0.0049,  0.0034,  0.0076],
#          [-0.0005,  0.0048,  0.0014,  ...,  0.0052,  0.0003,  0.0004]],

        # y_pred1 = self.fm(X, feature_emb).sum(dim=1, keepdim=True) #一个是特征交叉，一个是原始特征
        y_pred1 = self.fm(X, feature_emb)
        y_pred2 = self.mlp(feature_emb.flatten(start_dim=1))
        y_pred_re = y_pred1 + y_pred2

        y_pred1 = self.fm_bn(y_pred1)
        

        y_pred = self.output_activation(y_pred_re)  #nn.sigmoid(y_pred)

    
        return_dict = {"y_pred": y_pred, "y_pred1": y_pred1, "y_pred2": y_pred2}


        return return_dict

    def gene_data2(self):
        tag_ids = []
        user_id = torch.randint(low=1, high=self.user_vocab_len-2, size=(1024,), dtype=torch.float64)
        item_id = torch.randint(low=1, high=self.item_vocab_len-2, size=(1024,), dtype=torch.float64)
        
        for random_item_id in item_id:
            item = self.item_vocab_reversed.get(int(random_item_id))
            item = int(float(item))
            if item is not None:
                tags = self.item_tag.get(str(item))
                if tags:
                    random_tag = str(float(random.choice(tags)))
                    tag_id = self.tag_vocab[random_tag]
                    tag_ids.append(tag_id)
        tag_id = torch.tensor(tag_ids, dtype=torch.float64)
        # print(tag_id.shape) [1024]
        # exit()
        inputs_aug = torch.cat([user_id.unsqueeze(1), item_id.unsqueeze(1), tag_id.unsqueeze(1)], dim=1)
        # print(inputs_aug.shape) #torch.Size([1024, 3])
        # print(inputs_aug[:10])
        return inputs_aug

#ckt loss
    def calculate_loss(self, inputs): 
        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        # alpha = 0.6
        # temp = 0.07
        # y_pred1= return_dict["y_pred1"] #shape:[batch_size, 1]
        # y_pred2= return_dict["y_pred2"]   #shape:[batch_size, 1]
        # pos_indices = torch.nonzero(y_true == 1)[:, 0]
        # neg_indices = torch.nonzero(y_true == 0)[:, 0]

        # loss_cl = 0

        # for i in pos_indices:
        #     sim_pos = (torch.dot(y_pred1[i], y_pred2[i])) / temp
        #     sim_pos_e = torch.exp(sim_pos)
        #     sim_neg_e = 0
        #     for j in neg_indices:
        #         sim_neg = (torch.dot(y_pred1[i], y_pred1[j])) / temp  
        #         sim_neg_e += torch.exp(sim_neg)
        #     log_e = torch.log(sim_pos_e / sim_neg_e)
        #     loss_cl += -log_e
    #

    #新的一个版本
        temp = 0.05
        alpha = 1e-7


        y_pred1 = self.output_activation(return_dict["y_pred1"])
        y_pred2 = self.output_activation(return_dict["y_pred2"])


# lallal
        # pos_indices = torch.nonzero(y_true == 1)[:, 0]
        # neg_indices = torch.nonzero(y_true == 0)[:, 0]

        # # print(y_pred1[pos_indices].shape)
        # y_pred1_pos = y_pred1[pos_indices]
        # y_pred2_pos = y_pred2[pos_indices]
        # y_pred1_neg = y_pred1[neg_indices]
        # # print(y_pred1_pos.shape,y_pred2_pos.shape,y_pred1_neg.shape)

        # sim_pos = torch.sum(y_pred1_pos * y_pred2_pos, dim=1) / temp
        # sim_pos_e = torch.exp(sim_pos)

        # # Broadcasting: [batch_size_pos, 1] x [1, batch_size_neg]
        # sim_neg = torch.mm(y_pred1_pos, y_pred1_neg.t()) / temp
        # sim_neg_e = torch.exp(sim_neg).sum(dim=1)

        # # Element-wise division and log
        # loss_cl = -torch.log(sim_pos_e / sim_neg_e).sum()


#lallal

        pos_indices = torch.nonzero(y_true == 1)[:, 0]
        neg_indices = torch.nonzero(y_true == 0)[:, 0]
        # print(y_pred1[pos_indices].shape)

        sim_pos = torch.sum(y_pred1[pos_indices] * y_pred2[pos_indices], dim=1) / temp
        sim_pos_e = torch.exp(sim_pos).unsqueeze(1)

        sim_neg = torch.sum(y_pred1[pos_indices].unsqueeze(1) * y_pred1[neg_indices], dim=2) / temp
        sim_neg_e = torch.exp(sim_neg).sum(dim=1)

        log_e = torch.log(sim_pos_e / sim_neg_e)

        loss_cl = -torch.sum(log_e)
    # 又是一个版本  

        # temp = 0.05
        # alpha = 1e-8
        # alpha2 =1e-6
        # y_pred1 = self.output_activation(return_dict["y_pred1"])
        # y_pred2 = self.output_activation(return_dict["y_pred2"])
        # # y_pred2 = return_dict["y_pred2"]  
        # pos_indices = torch.nonzero(y_true == 1)[:, 0]
        # neg_indices = torch.nonzero(y_true == 0)[:, 0]

        # # print(y_pred1.shape,pos_indices.shape,neg_indices.shape) #[4096, 1]) torch.Size([1409, 2]) torch.Size([2687, 2])
        # # y_pred1_nonzero = y_pred1[pos_indices]
        # # # 确保维度正确
        # # y_pred1_nonzero = y_pred1_nonzero.reshape(-1, 1)
        # # print(y_pred1_nonzero.shape,y_pred2[pos_indices].shape,y_pred1[neg_indices].shape) #[1409,2,1]

        # #计算正样本与负样本之间的相似度
        # sim_pos1 = torch.matmul(y_pred1[pos_indices], y_pred2[pos_indices].T) / temp
        # sim_neg1 = torch.matmul(y_pred1[pos_indices], y_pred1[neg_indices].T) / temp

        # # 计算相似度的指数和
        # sim_pos_e1 = torch.exp(sim_pos1)
        # sim_neg_e1 = torch.exp(sim_neg1).sum(dim=1)  # 对负样本相似度进行求和

        # # 计算损失
        # log_e1 = torch.log(sim_pos_e1 / sim_neg_e1)
        # # print(sim_pos_e1.shape,sim_neg_e1.shape,log_e1.shape)

        # #计算正样本与负样本之间的相似度
        # # sim_pos2 = torch.matmul(y_pred2[pos_indices], y_pred1[pos_indices].T) / temp
        # # sim_neg2 = torch.matmul(y_pred2[pos_indices], y_pred2[neg_indices].T) / temp

        # # # 计算相似度的指数和
        # # sim_pos_e2 = torch.exp(sim_pos2)
        # # sim_neg_e2 = torch.exp(sim_neg2).sum(dim=1)  # 对负样本相似度进行求和

        # # # 计算损失
        # # log_e2 = torch.log(sim_pos_e2 / sim_neg_e2)

        # loss_cl = -log_e1.sum() 
        # # loss_cl2 = -log_e2.sum() 



        loss = self.loss_fn(return_dict['y_pred'], y_true,reduction='mean') +alpha*loss_cl 


        return loss


    def compute_cl_loss(self,return_dict):

        #欧几里得距离
        alpha = 1.2
        # y_pred1 = return_dict["y_pred1"]
        # y_pred2 = return_dict["y_pred2"]
        # y_pred = return_dict["y_pred_re"]
        # y_pred1 = self.projector1(y_pred1)
        # y_pred2 = self.projector2(y_pred2)
        # y_pred = self.projector3(y_pred)
        
        # # mse_loss = F.mse_loss(y_pred1, 1.2*y_pred2)
        # cl_loss1 = torch.norm(y_pred1.sub(y_pred), dim=1).pow(2).mean() #不要使用pow_2，是in_place操作，导致梯度出现问题
        # cl_loss2 = torch.norm(y_pred2.sub(y_pred), dim=1).pow(2).mean() #不要使用pow_2，是in_place操作，导致梯度出现问题
        # # print(cl_loss.shape) #torch.Size([])
        # cl_loss3 = torch.norm(y_pred1.sub(y_pred2), dim=1).pow(2).mean()  #对应每个dim上的拉近操作


        #余弦相似度的对比学习
        y_pred1 = return_dict["y_pred1"]
        y_pred2 = return_dict["y_pred2"]
        y_pred1 = self.projector1(y_pred1)
        y_pred2 = self.projector2(y_pred2)

        y_pred1 = y_pred1.view(-1)
        y_pred2 = y_pred2.view(-1)
        # print(a.shape,b.shape)
        cos_loss = 1 - F.cosine_similarity(y_pred1, y_pred2, dim=0)


        return  alpha *cos_loss




    def compute_ctrl_loss(self,inputs):
        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        temp = 0.05
        alpha = 1.5
        y_pred1 = return_dict["y_pred1"]
        y_pred2 = return_dict["y_pred2"]
        y_pred1 = self.projector1(y_pred1)
        y_pred2 = self.projector2(y_pred2)
        


        y_pred1 = self.fm_bn(y_pred1)
        y_pred2 = self.mlp_bn(y_pred2)

        # y_pred1 = self.output_activation(y_pred1)
        # y_pred2 = self.output_activation(y_pred2)

        y_pred1 = self.act(y_pred1)
        y_pred2 = self.act(y_pred2)

        # y_pred1_np = y_pred1.detach().cpu().numpy()
        # y_pred2_np = y_pred2.detach().cpu().numpy()

        # # 创建包含 y_pred1 和 y_pred2 的 dataframe
        # data = pd.DataFrame(np.concatenate([y_pred1_np, y_pred2_np], axis=1))

        # # 将dataframe保存为 CSV 文件
        # data.to_csv('deepbn2_ctrl3.csv', index=False)
# 一些奇怪的东西
        # return_dict = self.forward(inputs)
        # y_true = self.get_labels(inputs)

        # temp = 0.05
        # alpha = 0.1
        # y_pred1 = return_dict["y_pred1"]
        # y_pred2 = return_dict["y_pred2"]
        # y_pred1 = self.projector1(y_pred1)
        # y_pred2 = self.projector2(y_pred2)

        # y_pred1 = self.output_activation(y_pred1)
        # y_pred2 = self.output_activation(y_pred2)

        # y_pred1 = self.fm_ln(y_pred1)
        # y_pred2 = self.mlp_ln(y_pred2)

        # y_pred1 = self.fm_bn(y_pred1)
        # y_pred2 = self.mlp_bn(y_pred2)

        # loss_cl1 = 0
        # loss_cl2 = 0

        # for i in range(y_pred1.shape[0]):
        #     sim_pos = self.cosine_similarity(y_pred1[i], y_pred2[i]) / temp
        #     sim_neg = 0
        #     for j in range(y_pred1.shape[0]):
        #         sim_neg = torch.exp(self.cosine_similarity(y_pred1[i], y_pred2[j]) / temp) +sim_neg

        #     log_e = torch.log(sim_pos / sim_neg)

        #     loss_cl1 += -log_e


        # for i in range(y_pred2.shape[0]):
        #     sim_pos = self.cosine_similarity(y_pred2[i], y_pred1[i]) / temp
        #     sim_neg = 0
        #     for j in range(y_pred2.shape[0]):
        #         sim_neg = torch.exp(self.cosine_similarity(y_pred2[i], y_pred1[j]) / temp) +sim_neg

        #     log_e = torch.log(sim_pos / sim_neg)

        #     loss_cl2 += -log_e

        # loss_cl2 = loss_cl2/y_pred1.shape[0]

        # loss_cl = 1/2 *(loss_cl1 + loss_cl2)


        # loss_cl1 = 0
        # loss_cl2 = 0

        # sim_all1 = F.cosine_similarity(y_pred1, y_pred2, dim=1) / temp
        # sim_all2 = F.cosine_similarity(y_pred2, y_pred1, dim=1) / temp


        # for i in range(y_pred1.shape[0]):
        #     sim_pos = torch.exp(sim_all1[i])
        #     sim_neg = torch.sum(torch.exp(sim_all1))

        #     log_e = torch.log(sim_pos / sim_neg)

        #     loss_cl1 += -log_e

        # for i in range(y_pred2.shape[0]):
        #     sim_pos = torch.exp(sim_all2[i])
        #     sim_neg = torch.sum(torch.exp(sim_all2))

        #     log_e = torch.log(sim_pos / sim_neg)

        #     loss_cl2 += -log_e


        # loss_cl1 = torch.mean(loss_cl1)
        # loss_cl2 = torch.mean(loss_cl2)

        # loss_cl = 1/2 *(loss_cl1 + loss_cl2)

        # loss =self.loss_fn(return_dict['y_pred'], y_true,reduction='mean') +alpha*loss_cl 
        sim_all1 = torch.exp(F.cosine_similarity(y_pred1, y_pred2, dim=1) / temp) #dim=0对每列特征做相似度，dim=1对每行样本做相似度
        sim_all2 = torch.exp(F.cosine_similarity(y_pred2, y_pred1, dim=1) / temp)

        sim_neg1 = torch.sum(sim_all1)
        sim_neg2 = torch.sum(sim_all2)

        log_e1 = torch.log(sim_all1 / sim_neg1)
        log_e2 = torch.log(sim_all2 / sim_neg2)

        loss_cl1 = -torch.mean(log_e1)
        loss_cl2 = -torch.mean(log_e2)

        loss_cl = 1/2 *(loss_cl1 + loss_cl2)

        loss = self.loss_fn(return_dict['y_pred'], y_true, reduction='mean') + alpha * loss_cl

        return  loss




    def add_loss(self, inputs):

        alpha_e = self.alpha_e
        alpha_i = self.alpha_i

        alpha_i_aug = self.alpha_i_aug
        alpha_e_aug = self.alpha_e_aug

        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')


        y_pred1 = return_dict["y_pred1"]
        y_pred2 = return_dict["y_pred2"]
        y_pred1 = self.output_activation(y_pred1)
        y_pred2 = self.output_activation(y_pred2)

        y_pred = return_dict['y_pred']

    
        # loss1 = self.listmle(y_pred1, y_pred.detach())
        # loss2 = self.listmle(y_pred2, y_pred.detach())
        loss1 = self.loss_fn(y_pred1, y_pred.detach(), reduction='mean')
        loss2 = self.loss_fn(y_pred2, y_pred.detach(), reduction='mean')
        # loss = loss + alpha_e* loss1/(loss1/loss).detach() +  alpha_i * loss2/(loss2/loss).detach()   
      
        # loss =  alpha_e * loss1 + alpha_i * loss2 + loss  
    
        return loss
    

    def CosineSimilarityLoss(self, embedding_1, embedding_2):
        embedding_1 = torch.nn.functional.normalize(embedding_1) #做了一个归一化处理
        embedding_2 = torch.nn.functional.normalize(embedding_2)

        cosine_sim = F.cosine_similarity(embedding_1, embedding_2)
        cosine_similarity_loss = 1.0 - cosine_sim
        return torch.mean(cosine_similarity_loss)


    def InfoNCE(self, embedding_1, embedding_2, cl_temperature):
        embedding_1 = torch.nn.functional.normalize(embedding_1)
        embedding_2 = torch.nn.functional.normalize(embedding_2)

        pos_score = torch.exp(torch.tensor(1.0) / cl_temperature)

        ttl_score = torch.matmul(embedding_1, embedding_2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / cl_temperature).sum(dim=1)

        loss = - torch.log(pos_score / ttl_score + 10e-6)
        return torch.mean(loss)

    def sifo_nce(self, embedding_1, embedding_2, cl_temperature):
        embedding_1 = torch.nn.functional.normalize(embedding_1)
        embedding_2 = torch.nn.functional.normalize(embedding_2)
        
        batch_size = embedding_1.shape[0]

        sim_pos = (torch.ones(batch_size) / cl_temperature).to(embedding_1.device)
        sim_neg = torch.matmul(embedding_1, embedding_2.t())  # 计算所有正样本与所有负样本之间的相似度
        sim_neg_e = torch.sin((torch.pi / 4) * sim_neg - (torch.pi / 4)) + 1
        log_e = sim_pos / (sim_neg_e/cl_temperature).sum(dim=1)
        loss_cl =  log_e.sum()


        return torch.mean(loss_cl)


    def add_to_loss(self,return_dict,y_true):
        temp = 0.15
        alpha = 0.2
        temp_cos1 = 0.9
        temp_cos2 = 0.7
        y_pred1 = return_dict["y_pred1"]
        y_pred2 = return_dict["y_pred2"]
        y_pred1 = self.projector1(y_pred1)
        y_pred2 = self.projector2(y_pred2)
        # y_pred =return_dict['y_pred']
        # y_pred = self.projector3(y_pred)
        info_loss = self.InfoNCE(y_pred1, y_pred2, temp)
        cl12 = self.CosineSimilarityLoss(y_pred1, y_pred2)
        # cl13 = self.CosineSimilarityLoss(y_pred1, y_pred)
        # cl23 = self.CosineSimilarityLoss(y_pred2, y_pred)

        loss = self.loss_fn(return_dict['y_pred'], y_true, reduction='mean') + alpha * info_loss + temp_cos1 * cl12


        return loss
    

    def sin_loss(self,return_dict,y_true):
        temp = 0.15
        alpha = 0.3
        temp_cos1 = 1.0
        y_pred1 = return_dict["y_pred1"]
        y_pred2 = return_dict["y_pred2"]
        y_pred1 = self.projector1(y_pred1)
        y_pred2 = self.projector2(y_pred2)
        # y_pred =return_dict['y_pred']
        # y_pred = self.projector3(y_pred)
        info_loss = self.sifo_nce(y_pred1, y_pred2, temp)
        cl12 = self.CosineSimilarityLoss(y_pred1, y_pred2)
        # cl13 = self.CosineSimilarityLoss(y_pred1, y_pred)
        # cl23 = self.CosineSimilarityLoss(y_pred2, y_pred)

        loss = self.loss_fn(return_dict['y_pred'], y_true, reduction='mean') + alpha * info_loss + temp_cos1 * cl12




        return loss

    def forward1(self, inputs):

        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        y_pred1 = self.fm(X, feature_emb)
        y_pred2= self.mlp(feature_emb.flatten(start_dim=1))
        y_pred =    y_pred1 + y_pred2

        
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict


    # def forward_train(self, inputs):
    #     """
    #     Inputs: [X,y]
    #     """
    #     X = self.get_inputs(inputs)
    #     feature_emb = self.embedding_layer(X) 
    #     y_pred1 = self.fm(X, feature_emb) #一个是特征交叉，一个是原始特征
    
    #     y_pred2 = self.mlp(feature_emb.flatten(start_dim=1)) #mlp初始化了output_dim=1,输入是[batch_size, feature_num*embedding_dim]

    #     y_pred =    y_pred1 + y_pred2
    #     y_pred = self.output_activation(y_pred)  #nn.sigmoid(y_pred)
    #     return_dict = {"y_pred": y_pred}


    #     return return_dict #[batch_size,1]
    
    # def forward_valid(self, inputs):
    #     """
    #     Inputs: [X,y]
    #     """
    #     X = self.get_inputs(inputs)
    #     feature_emb = self.embedding_layer(X) 
        # y_pred1 = self.fm(X, feature_emb) #一个是特征交叉，一个是原始特征
    
        # y_pred2 = self.mlp(feature_emb.flatten(start_dim=1)) #mlp初始化了output_dim=1,输入是[batch_size, feature_num*embedding_dim]

        # y_pred =    y_pred1 + y_pred2
        # y_pred = self.output_activation(y_pred)  #nn.sigmoid(y_pred)
        # return_dict = {"y_pred": y_pred}

    #     return return_dict


    def list_loss(self,preds,labels):
        eps=1e-7
        # loss_1 = -1 * labels * torch.log(preds+eps)
        # loss_0 = -1 *(1-labels) * torch.log(1-preds+eps)
        # loss = loss_1 + loss_0
        loss_1 = -1 * labels * torch.log(preds)
        loss_0 = -1 *(1-labels) * torch.log(1-preds)
        loss = loss_1 + loss_0
        
        # loss = -1 * labels * torch.log(preds) *(1+ torch.pow((1-preds),2))
        # print(loss)
        # exit()
        # return torch.sum(loss)
        return torch.mean(loss)



class Focal_Loss():

    def __init__(self,alpha=0.25,gamma=2):
        super(Focal_Loss,self).__init__()
        self.alpha=alpha
        self.gamma=gamma
    
    def forward(self,preds,labels):
        """
        preds:sigmoid的输出结果
        labels：标签
        """
        eps=1e-7
        # loss_1=-1 * torch.pow((1-preds),self.gamma)*torch.log(preds+eps)*labels
        loss_1=-1*self.alpha*torch.pow((1-preds),self.gamma)*torch.log(preds+eps)*labels
        loss_0=-1*(1-self.alpha)*torch.pow(preds,self.gamma)*torch.log(1-preds+eps)*(1-labels)
        # loss_0=-1 * torch.pow(preds,self.gamma)*torch.log(1-preds+eps)*(1-labels)
        loss=loss_0 + loss_1
        return torch.mean(loss)
    def __call__(self, preds,labels):
        return self.forward(preds,labels)
    

class Focal_Loss_two():

    def __init__(self,alpha=0.5,gamma=2):
        super(Focal_Loss_two,self).__init__()
        self.alpha=alpha
        self.gamma=gamma
    
    def forward(self,preds,labels):
        """
        preds:sigmoid的输出结果
        labels：标签
        """
        eps=1e-7
        # loss_1=-1 * torch.pow((1-preds),self.gamma)*torch.log(preds+eps)*labels
        #通过加一改变这个值
        # loss_1=-1*self.alpha*(1 + torch.pow((1-preds),self.gamma))*torch.log(preds+eps)*labels
        # loss_0=-1*(1-self.alpha)*(1 + torch.pow(preds,self.gamma))*torch.log(1-preds+eps)*(1-labels)

        #通过torch.exp来改变
        loss_1=-1*self.alpha*torch.exp(1-preds)*torch.log(preds+eps)*labels
        loss_0=-1*(1-self.alpha)*torch.exp(preds)*torch.log(1-preds+eps)*(1-labels)

        # loss_0=-1 * torch.pow(preds,self.gamma)*torch.log(1-preds+eps)*(1-labels)
        loss=loss_0 + loss_1
        return torch.mean(loss)
    def __call__(self, preds,labels):
        return self.forward(preds,labels)



'''上面是deepfm的代码，由于导包导不进去，现将KDDeepFM复制过来'''

# import torch
# from torch import nn
# from fuxictr.pytorch.models import BaseModel
# from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, FactorizationMachine,MLP_Split_Block
# import torch.nn.functional as F

# class KDDeepFM(BaseModel):
#     def __init__(self, 
#                  feature_map, 
#                  model_id="KDDeepFM", 
#                  gpu=-1, 
#                  learning_rate=1e-3, 
#                  embedding_dim=10, 
#                  hidden_units=[64, 64, 64], 
#                  student_hidden_units=[1024, 1024, 1024],
#                  hidden_activations="ReLU", 
#                  net_dropout=0, 
#                  batch_norm=False, 
#                  embedding_regularizer=None, 
#                  net_regularizer=None,
#                  alpha_e = 1,
#                  alpha_i = 1,
#                  phase="teacher_training",
#                  **kwargs):
#         super(KDDeepFM, self).__init__(feature_map, 
#                                      model_id=model_id, 
#                                      gpu=gpu, 
#                                      embedding_regularizer=embedding_regularizer, 
#                                      net_regularizer=net_regularizer,
#                                      **kwargs)
#         self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
#         self.embedding_layer_mlp = FeatureEmbedding(feature_map, embedding_dim)
#         input_dim = feature_map.sum_emb_out_dim()
#         self.phase = phase
#         self.alpha_e = alpha_e
#         self.alpha_i = alpha_i
#         self.teacher_net = DeepFM(feature_map,hidden_units,hidden_activations,net_dropout,batch_norm)


#         self.student_net = MLP_Block(input_dim=input_dim,
#                                         output_dim=None, # output hidden layer
#                                         hidden_units=student_hidden_units,
#                                         hidden_activations=hidden_activations,
#                                         output_activation=None,
#                                         dropout_rates=net_dropout,
#                                         batch_norm=batch_norm)

#         self.parallel_dnn = MLP_Block(input_dim=input_dim,
#                                         output_dim=None, # output hidden layer
#                                         hidden_units=hidden_units,
#                                         hidden_activations=hidden_activations,
#                                         output_activation=None,
#                                         dropout_rates=net_dropout,
#                                         batch_norm=batch_norm)
#         # final_dim = input_dim 
#         # self.fc_cross = nn.Linear(final_dim, 1)
#         self.fc_mlp = nn.Linear(hidden_units[-1], 1)
#         self.fc_student = nn.Linear(student_hidden_units[-1], 1)
        
#         print("now is KDDeepFM")

#         self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
#         self.reset_parameters()
#         self.model_to_device()

#         student_file = "./Movielens/KDDeepFM/student/movielenslatest_x1_cd32d937/KDDeepFM_movielenslatest.model"
#         teacher_file = "./Avazu/KDDeepFM/teacher/avazu_x1_3fb65689/KDDeepFM_avazu_x1.model"
#         if self.phase == "distillation":
#             save_info = torch.load(teacher_file)
#             self.load_state_dict(save_info)
#             print("load teacher model")
#         elif self.phase == "finetuning":
#             save_info = torch.load(student_file)
#             self.load_state_dict(save_info)
#             print("load student model")


#         # if self.phase != "teacher_training":
#         #     save_info = torch.load("./Movielens/DCN_movielenslatest_x1/KGDCNv2/teacher/movielenslatest_x1_cd32d937/KGDCNv2_movielenslatest.model")
#         #     self.load_state_dict(save_info)



#     def forward(self, inputs):
#         X = self.get_inputs(inputs)

#         teac_stud_embeddings = self.embedding_layer(X, flatten_emb=True)


#         if self.phase == "teacher_training":
#             y_pred = self.teacher_net(X,teac_stud_embeddings)
#             return_dict = {"y_pred": y_pred}
#             return return_dict

#         elif self.phase == "distillation":
#             mlp_embedding = self.embedding_layer_mlp(X, flatten_emb=True)
#             # print(mlp_embedding.shape)
#             # exit()
#             # teac_stud_embeddings = teac_stud_embeddings.data
#             if self.training:
#                 with torch.no_grad():
#                     t_pred = self.teacher_net(X,teac_stud_embeddings)

#             # with torch.no_grad():
#             #     y_pred2 = self.parallel_dnn(mlp_embedding)
#             #     y_pred2 =self.fc_mlp(y_pred2)
#             y_pred1 = self.student_net(teac_stud_embeddings)
#             y_pred2 = self.parallel_dnn(mlp_embedding)
#             y_pred1 = self.fc_student(y_pred1)
#             y_pred2 = self.fc_mlp(y_pred2)
     
            
#             s_pred = 0.5*(y_pred1 + y_pred2)
#             s_pred = self.output_activation(s_pred)
#             return_dict = {"y_pred1": y_pred1, "y_pred2": y_pred2, "y_pred": s_pred}
#             return return_dict
        
#         elif self.phase == "finetuning":
#             mlp_embedding = self.embedding_layer_mlp(X, flatten_emb=True)
#             y_pred1 = self.student_net(teac_stud_embeddings)
#             y_pred2 = self.parallel_dnn(mlp_embedding)
#             y_pred1 = self.fc_student(y_pred1)
#             y_pred2 = self.fc_mlp(y_pred2)
#             s_pred = 0.5*(y_pred1 + y_pred2)
#             s_pred = self.output_activation(s_pred)
#             return_dict = { "y_pred": s_pred, "y_pred1": y_pred1, "y_pred2": y_pred2}
#             return return_dict

#         else:
#             raise ValueError("Invalid phase")


#     # def FeatureInteraction(self, data):
#     #     if self.phase == "teacher_training":
#     #         return self.teacher_net.FeatureInteraction(data)
#     #     else:
#     #         return self.student_net.FeatureInteraction(data)


#     def add_loss(self, inputs):

#         alpha_e = self.alpha_e
#         alpha_i = self.alpha_i     

#         loss = None
#         # return_dict = self.forward(inputs)
#         y_true = self.get_labels(inputs)

#         if self.phase == "teacher_training" :
#             return_dict = self.forward(inputs)
#             y_pred = return_dict['y_pred']
#             # print(y_pred.shape)
#             # print(y_true.shape)
#             # exit()
#             loss = self.loss_fn(y_pred, y_true, reduction='mean')
#         elif self.phase == "distillation":
#             self.teacher_net.eval()
#             return_dict = self.forward(inputs)
#             y_pred1 = return_dict['y_pred1']  # student network output
#             y_pred2 = return_dict['y_pred2']  #mlp output
#             y_pred = return_dict['y_pred']  # final output
#             ctr_loss = self.loss_fn(y_pred, y_true, reduction='mean') #ctr loss
#             kd_loss = torch.mean(self.teacher_net.logits.data - y_pred1)**2

#             # y1 = self.output_activation(y_pred1)
#             # y2 = self.output_activation(y_pred2)

#             # loss1 = self.loss_fn(y1,y_pred.detach(), reduction='mean')
#             # loss2 = self.loss_fn(y2,y_pred.detach(), reduction='mean')
#             loss = kd_loss + ctr_loss

#             # loss = alpha_e * loss1 + alpha_i * loss2 + kd_loss + ctr_loss
#         elif self.phase == "finetuning":
#             return_dict = self.forward(inputs)
#             y_pred1 = return_dict['y_pred1']  # student network output
#             y_pred2 = return_dict['y_pred2']  #mlp output
#             y_pred = return_dict['y_pred']  # final output
#             ctr_loss = self.loss_fn(y_pred, y_true, reduction='mean') #ctr loss
#             y1 = self.output_activation(y_pred1)
#             y2 = self.output_activation(y_pred2)
#             loss1 = self.loss_fn(y1,y_true, reduction='mean')
#             loss2 = self.loss_fn(y2,y_true, reduction='mean')
#             loss = alpha_e * loss1 + alpha_i * loss2 + ctr_loss



#         if loss is None:
#             raise ValueError("Loss has not been assigned any value.")

#         return loss


# class DeepFM(nn.Module):
#     def __init__(self, 
#                  feature_map, 
#                  hidden_units=[64, 64, 64], 
#                  hidden_activations="ReLU", 
#                  net_dropout=0, 
#                  batch_norm=False, 
#                  ):
#         super(DeepFM, self).__init__()
#         self.output_activation = nn.Sigmoid()
#         self.fm = FactorizationMachine(feature_map)
#         self.mlp = MLP_Block(input_dim=feature_map.sum_emb_out_dim(),
#                              output_dim=1, 
#                              hidden_units=hidden_units,
#                              hidden_activations=hidden_activations,
#                              output_activation=None, 
#                              dropout_rates=net_dropout, 
#                              batch_norm=batch_norm)
        
#     def FeaureInteraction(self,X,feature_emb):
#         fm_out = self.fm(X,feature_emb)
#         mlp_out = self.mlp(feature_emb.flatten(start_dim=1))
#         self.logits = fm_out + mlp_out
#         self.final_out = self.output_activation(self.logits)
#         return self.final_out
    
#     def forward(self, X, inputs):
#         return self.FeaureInteraction(X,inputs)




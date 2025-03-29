import json
import random
import pandas as pd
import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block



class DSMLP(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="DSMLP",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 dnn_hidden_units=[64, 64, 64],
                 parallel_dnn_hidden_units=[],
                 student_dnn_hidden_units=[],
                 dnn_activations="ReLU",
                 num_cross_layers=3,
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 phase = "teacher_training",
                 alpha_e = 1,
                 alpha_i = 1,
                 **kwargs):
        super(DSMLP, self).__init__(feature_map,
                                    model_id=model_id,
                                    gpu=gpu,
                                    embedding_regularizer=embedding_regularizer,
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.embedding_layer_mlp = FeatureEmbedding(feature_map, embedding_dim)
        input_dim = feature_map.sum_emb_out_dim()
        self.phase = phase
        self.alpha_e = alpha_e
        self.alpha_i = alpha_i
        self.teacher_net = GDCN(input_dim,dnn_hidden_units,dnn_activations, num_cross_layers,net_dropout,batch_norm)


        self.student_net = MLP_Block(input_dim=input_dim,
                                        output_dim=None, # output hidden layer
                                        hidden_units=student_dnn_hidden_units,
                                        hidden_activations=dnn_activations,
                                        output_activation=None,
                                        dropout_rates=net_dropout,
                                        batch_norm=batch_norm)

        self.parallel_dnn = MLP_Block(input_dim=input_dim,
                                        output_dim=None, # output hidden layer
                                        hidden_units=parallel_dnn_hidden_units,
                                        hidden_activations=dnn_activations,
                                        output_activation=None,
                                        dropout_rates=net_dropout,
                                        batch_norm=batch_norm)
        self.cross_bn = nn.BatchNorm1d(student_dnn_hidden_units[-1])
        self.dnn_bn = nn.BatchNorm1d(parallel_dnn_hidden_units[-1])

        self.fc_mlp = nn.Linear(parallel_dnn_hidden_units[-1], 1)
        self.fc_student = nn.Linear(student_dnn_hidden_units[-1], 1)
        
        print("now is DSMLP")

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        

        self.model_to_device()

        student_file = "./Criteo/DSMLP/student/xxx.model"
        teacher_file = "./Criteo/DSMLP/teacher/xxx.model"


        if self.phase == "distillation":
            save_info = torch.load(teacher_file)
            self.load_state_dict(save_info)
            print("load teacher model")
        elif self.phase == "finetuning":
            save_info = torch.load(student_file)
            self.load_state_dict(save_info)
            print("load student model")
        

    def forward(self, inputs):
        X = self.get_inputs(inputs)

        teac_stud_embeddings = self.embedding_layer(X, flatten_emb=True)


        if self.phase == "teacher_training":
            y_pred = self.teacher_net(teac_stud_embeddings)
            return_dict = {"y_pred": y_pred}
            return return_dict

        elif self.phase == "distillation":
            mlp_embedding = self.embedding_layer_mlp(X, flatten_emb=True)

            if self.training:
                with torch.no_grad():
                    t_pred = self.teacher_net(teac_stud_embeddings)

            y_pred1 = self.student_net(teac_stud_embeddings)
            y_pred2 = self.parallel_dnn(mlp_embedding)

            y_pred1 = self.cross_bn(y_pred1)
            y_pred2 = self.dnn_bn(y_pred2)

            y_pred1 = self.fc_student(y_pred1)
            y_pred2 = self.fc_mlp(y_pred2)
     
            
            s_pred = 0.5*(y_pred1 + y_pred2)
            s_pred = self.output_activation(s_pred)
            return_dict = {"y_pred1": y_pred1, "y_pred2": y_pred2, "y_pred": s_pred}
            return return_dict
        
        elif self.phase == "finetuning":
            mlp_embedding = self.embedding_layer_mlp(X, flatten_emb=True)
            y_pred1 = self.student_net(teac_stud_embeddings)
            y_pred2 = self.parallel_dnn(mlp_embedding)
            y_pred1 = self.fc_student(y_pred1)
            y_pred2 = self.fc_mlp(y_pred2)
            s_pred = 0.5*(y_pred1 + y_pred2)
            s_pred = self.output_activation(s_pred)
            return_dict = { "y_pred": s_pred, "y_pred1": y_pred1, "y_pred2": y_pred2}
            return return_dict

        else:
            raise ValueError("Invalid phase")


    def FeatureInteraction(self, data):
        if self.phase == "teacher_training":
            return self.teacher_net.FeatureInteraction(data)
        else:
            return self.student_net.FeatureInteraction(data)


    def add_loss(self, inputs):

        alpha_e = self.alpha_e
        alpha_i = self.alpha_i     

        loss = None
        y_true = self.get_labels(inputs)

        if self.phase == "teacher_training" :
            return_dict = self.forward(inputs)
            y_pred = return_dict['y_pred']
            loss = self.loss_fn(y_pred, y_true, reduction='mean')
        elif self.phase == "distillation":
            self.teacher_net.eval()
            return_dict = self.forward(inputs)
            y_pred1 = return_dict['y_pred1']  # student network output
            y_pred2 = return_dict['y_pred2']  #mlp output
            y_pred = return_dict['y_pred']  # final output
            y_pred1 = self.output_activation(y_pred1)
            y_pred2 = self.output_activation(y_pred2)
            ctr_loss = self.loss_fn(y_pred, y_true, reduction='mean') #ctr loss
            kd_loss = torch.mean(self.teacher_net.outputs.data - y_pred1)**2
            loss1 = self.loss_fn(y_pred1,y_true, reduction='mean')
            loss2 = self.loss_fn(y_pred2,y_true, reduction='mean')


            loss = alpha_e * loss1 + alpha_i * loss2 + kd_loss + ctr_loss
        elif self.phase == "finetuning":
            return_dict = self.forward(inputs)
            y_pred1 = return_dict['y_pred1']  # student network output
            y_pred2 = return_dict['y_pred2']  #mlp output
            y_pred = return_dict['y_pred']  # final output
            ctr_loss = self.loss_fn(y_pred, y_true, reduction='mean') #ctr loss
            y1 = self.output_activation(y_pred1)
            y2 = self.output_activation(y_pred2)
            loss1 = self.loss_fn(y1,y_true, reduction='mean')
            loss2 = self.loss_fn(y2,y_true, reduction='mean')
            loss = alpha_e * loss1 + alpha_i * loss2 + ctr_loss


        if loss is None:
            raise ValueError("Loss has not been assigned any value.")

        return loss






class GDCN(nn.Module):
    def __init__(self,
                 input_dim,
                 parallel_dnn_hidden_units=[],
                 dnn_activations="ReLU",
                 num_cross_layers=3,
                 net_dropout=0,
                 batch_norm=False):
        super(GDCN, self).__init__()

        self.cross_net = GateCorssLayer(input_dim, num_cross_layers)

        # exit()
        self.parallel_dnn = MLP_Block(input_dim=input_dim,
                                        output_dim=None, # output hidden layer
                                        hidden_units=parallel_dnn_hidden_units,
                                        hidden_activations=dnn_activations,
                                        output_activation=None,
                                        dropout_rates=net_dropout,
                                        batch_norm=batch_norm)
        final_dim = input_dim + parallel_dnn_hidden_units[-1]
        self.fc = nn.Linear(final_dim, 1)
        self.output_activation = nn.Sigmoid()

    def FeatureInteraction(self, feature_emb):
        cross_out = self.cross_net(feature_emb)
        dnn_out = self.parallel_dnn(feature_emb)
        final_out = torch.cat([cross_out, dnn_out], dim=-1)
        self.logits = self.fc(final_out)
        self.outputs = self.output_activation(self.logits)

        return self.outputs
    
    def forward(self, feature):
        return self.FeatureInteraction(feature)



class GateCorssLayer(nn.Module):
    #  The core structure： gated corss layer.
    def __init__(self, input_dim, cn_layers=3):
        super().__init__()

        self.cn_layers = cn_layers

        self.w = nn.ModuleList([
            nn.Linear(input_dim, input_dim, bias=False) for _ in range(cn_layers)
        ])
        self.wg = nn.ModuleList([
            nn.Linear(input_dim, input_dim, bias=False) for _ in range(cn_layers)
        ])

        self.b = nn.ParameterList([nn.Parameter(
            torch.zeros((input_dim,))) for _ in range(cn_layers)])

        for i in range(cn_layers):
            nn.init.uniform_(self.b[i].data)

        self.activation = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        for i in range(self.cn_layers):
            xw = self.w[i](x) # Feature Crossing
            xg = self.activation(self.wg[i](x)) # Information Gate
            x = x0 * (xw + self.b[i]) * xg + x
        return x
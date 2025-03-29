import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, FactorizationMachine,MLP_Split_Block
import torch.nn.functional as F

class KDDFM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="KDDFM",
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 hidden_units=[64, 64, 64], 
                 student_hidden_units=[64, 64, 64],
                 parallel_hidden_units=[64, 64, 64],
                 hidden_activations="ReLU",
                 net_dropout=0, 
                 dropout_stu = 0,
                 batch_norm=False,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 phase = "teacher_training",
                 alpha_e = 1,
                 alpha_i = 1,
                 **kwargs):
        super(KDDFM, self).__init__(feature_map, 
                                         model_id=model_id, 
                                         gpu=gpu, 
                                         embedding_regularizer=embedding_regularizer, 
                                         net_regularizer=net_regularizer,
                                         **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.embedding_layer_mlp = FeatureEmbedding(feature_map, embedding_dim)
        input_dim = embedding_dim * feature_map.num_fields

        self.phase = phase
        self.alpha_e = alpha_e
        self.alpha_i = alpha_i 

        self.teacher_net = DFM(feature_map,
                               input_dim,
                               hidden_units,
                               hidden_activations,
                               net_dropout,
                               batch_norm)
                                    
        self.student_net =  MLP_Block(input_dim=input_dim,
                              output_dim=None, 
                              hidden_units=student_hidden_units,
                              hidden_activations=hidden_activations,
                              output_activation=None,
                              dropout_rates = dropout_stu,
                              batch_norm=batch_norm,
                              )

        self.parallel_dnn = MLP_Block(input_dim=input_dim,
                                        output_dim=None, # output hidden layer
                                        hidden_units=parallel_hidden_units,
                                        hidden_activations=hidden_activations,
                                        output_activation=None,
                                        dropout_rates=dropout_stu,
                                        batch_norm=batch_norm)
     
        self.fc_student = nn.Linear(student_hidden_units[-1], 1)
        self.fc_mlp = nn.Linear(parallel_hidden_units[-1], 1)
        self.bn_stu = nn.BatchNorm1d(student_hidden_units[-1])
        self.bn_mlp = nn.BatchNorm1d(parallel_hidden_units[-1])
        self.ln_stu = nn.LayerNorm(student_hidden_units[-1])
        self.ln_par = nn.LayerNorm(parallel_hidden_units[-1])

        print("now is KDDFM")

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

        student_file = "./Movielens/KDDFM/student/xxx.model"
        teacher_file = "./Criteo/KDFM/teacher/xxx.model"
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

        teac_embeddings = self.embedding_layer(X)
        stud_embeddings = self.embedding_layer(X, flatten_emb=True)

        if self.phase == "teacher_training":
            y_pred = self.teacher_net(X,teac_embeddings)
            return_dict = {"y_pred": y_pred}
            return return_dict

        elif self.phase == "distillation":
            mlp_embedding = self.embedding_layer_mlp(X, flatten_emb=True)

            if self.training:
                with torch.no_grad():
                    t_pred = self.teacher_net(X,teac_embeddings)

            y_pred1 = self.student_net(stud_embeddings)
            y_pred2 = self.parallel_dnn(mlp_embedding)
            y_pred1 = self.fc_student(y_pred1)
            y_pred2 = self.fc_mlp(y_pred2)
     
            
            s_pred = 0.5*(y_pred1 + y_pred2)
            s_pred = self.output_activation(s_pred)
            return_dict = {"y_pred1": y_pred1, "y_pred2": y_pred2, "y_pred": s_pred}
            return return_dict
        
        elif self.phase == "finetuning":
            mlp_embedding = self.embedding_layer_mlp(X, flatten_emb=True)
            y_pred1 = self.student_net(stud_embeddings)
            y_pred2 = self.parallel_dnn(mlp_embedding)
            y_pred1 = self.ln_stu(y_pred1)
            y_pred2 = self.ln_par(y_pred2)
            y_pred1 = self.bn_stu(y_pred1)
            y_pred2 = self.bn_mlp(y_pred2)
            y_pred1 = self.fc_student(y_pred1)
            y_pred2 = self.fc_mlp(y_pred2)
            s_pred = 0.5*(y_pred1 + y_pred2)
            s_pred = self.output_activation(s_pred)
            return_dict = { "y_pred": s_pred, "y_pred1": y_pred1, "y_pred2": y_pred2}
            return return_dict

        else:
            raise ValueError("Invalid phase")

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
            ctr_loss = self.loss_fn(y_pred, y_true, reduction='mean') #ctr loss
            kd_loss = torch.mean(self.teacher_net.outputs.data - y_pred1)**2

            loss = kd_loss + ctr_loss

            
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


class DFM(nn.Module):
    def __init__(self, 
                 feature_map, 
                 input_dim,
                 hidden_units=[64, 64, 64],
                 hidden_activations="ReLU",
                 dropout_rates=0,
                 batch_norm=False,
                 ):
        super(DFM, self).__init__()
        self.output_activation = nn.Sigmoid()
        self.fm = FactorizationMachine(feature_map)
        self.mlp = MLP_Block(
                input_dim = input_dim,
                output_dim = 1,
                hidden_units = hidden_units,
                hidden_activations = hidden_activations,
                output_activation = None,
                dropout_rates = dropout_rates,
                batch_norm = batch_norm
        )

    def FeaureInteraction(self,X,feature_emb):
        fm_out = self.fm(X,feature_emb)
        mlp_out = self.mlp(feature_emb.flatten(start_dim=1))
        y_pred = fm_out + mlp_out
        self.outputs = self.output_activation(y_pred)
        return self.outputs
    
    def forward(self, X, inputs):
        return self.FeaureInteraction(X,inputs)

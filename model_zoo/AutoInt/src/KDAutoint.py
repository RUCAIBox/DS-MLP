from torch import nn
import torch
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, ScaledDotProductAttention, LogisticRegression


class KDAutoInt(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="KDAutoInt",
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 dnn_hidden_units=[64, 64, 64], 
                 student_hidden_units=[64, 64, 64],
                 parallel_hidden_units=[64, 64, 64],
                 dnn_activations="ReLU",
                 attention_layers = 2 ,
                 num_heads=1,
                 attention_dim=8,
                 net_dropout=0, 
                 dropout_stu = 0,
                 batch_norm=False,
                 layer_norm=False,
                 use_scale=False,
                 use_wide=False,
                 use_residual=True,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 phase = "teacher_training",
                 alpha_e = 1,
                 alpha_i = 1,
                 **kwargs):
        super(KDAutoInt, self).__init__(feature_map, 
                                         model_id=model_id, 
                                         gpu=gpu, 
                                         embedding_regularizer=embedding_regularizer, 
                                         net_regularizer=net_regularizer,
                                         **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.embedding_layer_mlp = FeatureEmbedding(feature_map, embedding_dim)
        feature_dim = embedding_dim * feature_map.num_fields

        self.phase = phase
        self.alpha_e = alpha_e
        self.alpha_i = alpha_i 

        self.teacher_net = AutoInt(
                                    feature_map,
                                    embedding_dim,
                                    dnn_hidden_units,
                                    dnn_activations,
                                    attention_layers,
                                    num_heads,
                                    attention_dim,
                                    net_dropout,
                                    layer_norm,
                                    batch_norm,
                                    use_scale,
                                    use_wide,
                                    use_residual,
                                )
                                    
        self.student_net =  MLP_Block(input_dim=feature_dim,
                              output_dim=None, 
                              hidden_units=student_hidden_units,
                              hidden_activations=dnn_activations,
                              output_activation=None,
                              dropout_rates=dropout_stu,
                              batch_norm=batch_norm,
                              )

        self.parallel_dnn = MLP_Block(input_dim=feature_dim,
                                        output_dim=None, # output hidden layer
                                        hidden_units=parallel_hidden_units,
                                        hidden_activations=dnn_activations,
                                        output_activation=None,
                                        dropout_rates=dropout_stu,
                                        batch_norm=batch_norm)
        self.fc_student = nn.Linear(student_hidden_units[-1], 1)
        self.fc_mlp = nn.Linear(parallel_hidden_units[-1], 1)
        self.bn_stu = nn.BatchNorm1d(student_hidden_units[-1])
        self.bn_mlp = nn.BatchNorm1d(parallel_hidden_units[-1])
        
        print("now is KDAutoInt")

   
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

        student_file = "./Criteo/KDAutoInt/student/xxx.model"
        teacher_file = "./Movielens/KDAutoInt/teacher/xxx.model"
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
            y_pred1 = self.bn_stu(y_pred1)
            y_pred2 = self.bn_mlp(y_pred2)
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


class AutoInt(nn.Module):
    def __init__(self, 
                 feature_map, 
                 embedding_dim=10, 
                 dnn_hidden_units=[64, 64, 64], 
                 dnn_activations="ReLU", 
                 attention_layers=2,
                 num_heads=1,
                 attention_dim=8,
                 net_dropout=0, 
                 batch_norm=False,
                 layer_norm=False,
                 use_scale=False,
                 use_wide=False,
                 use_residual=True,
                 ):
        super(AutoInt, self).__init__() 
      
        self.lr_layer = LogisticRegression(feature_map, use_bias=False) if use_wide else None
        self.dnn = MLP_Block(input_dim=feature_map.sum_emb_out_dim(),
                             output_dim=1, 
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm) \
                   if dnn_hidden_units else None # in case no DNN used
        self.self_attention = nn.Sequential(
            *[MultiHeadSelfAttention(embedding_dim if i == 0 else attention_dim,
                                     attention_dim=attention_dim, 
                                     num_heads=num_heads, 
                                     dropout_rate=net_dropout, 
                                     use_residual=use_residual, 
                                     use_scale=use_scale,
                                     layer_norm=layer_norm) \
             for i in range(attention_layers)])
        self.fc = nn.Linear(feature_map.num_fields * attention_dim, 1)
        self.output_activation = nn.Sigmoid()

    def forward(self, X , feature_emb):
        """
        Inputs: [X, y]
        """
        attention_out = self.self_attention(feature_emb)
        attention_out = torch.flatten(attention_out, start_dim=1)
        y_pred = self.fc(attention_out)
        if self.dnn is not None:
            y_pred += self.dnn(feature_emb.flatten(start_dim=1))
        if self.lr_layer is not None:
            y_pred += self.lr_layer(X)
        self.outputs = self.output_activation(y_pred)
       
        return self.outputs
    


class MultiHeadSelfAttention(nn.Module):
    """ Multi-head attention module """

    def __init__(self, input_dim, attention_dim=None, num_heads=1, dropout_rate=0., 
                 use_residual=True, use_scale=False, layer_norm=False):
        super(MultiHeadSelfAttention, self).__init__()
        if attention_dim is None:
            attention_dim = input_dim
        assert attention_dim % num_heads == 0, \
               "attention_dim={} is not divisible by num_heads={}".format(attention_dim, num_heads)
        self.head_dim = attention_dim // num_heads
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.scale = self.head_dim ** 0.5 if use_scale else None
        self.W_q = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_k = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_v = nn.Linear(input_dim, attention_dim, bias=False)
        if self.use_residual and input_dim != attention_dim:
            self.W_res = nn.Linear(input_dim, attention_dim, bias=False)
        else:
            self.W_res = None
        self.dot_attention = ScaledDotProductAttention(dropout_rate)
        self.layer_norm = nn.LayerNorm(attention_dim) if layer_norm else None

    def forward(self, X):
        residual = X
        
        # linear projection
        query = self.W_q(X)
        key = self.W_k(X)
        value = self.W_v(X)
        
        # split by heads
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot product attention
        output, attention = self.dot_attention(query, key, value, scale=self.scale)
        # concat heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        if self.W_res is not None:
            residual = self.W_res(residual)
        if self.use_residual:
            output += residual
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        output = output.relu()
        return output



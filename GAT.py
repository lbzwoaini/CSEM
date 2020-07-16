import numpy as np
import tensorflow as tf

import layers
from base_gattn import BaseGAttN

class GAT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, attn_drop, ffd_drop, bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        attns = []
        for _ in range(n_heads[0]):
            attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                out_sz=hid_units[0], activation=activation,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        #一个节点的K个注意力输出的连接
        h_1 = tf.concat(attns, axis=-1)
        
        #中间层定义
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
                    out_sz=hid_units[i], activation=activation,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
            h_1 = tf.concat(attns, axis=-1)
        out = []
        
        #最后一层的结构，均值输出
        for i in range(n_heads[-1]):
            out.append(layers.attn_head(h_1, bias_mat=bias_mat,
                out_sz=nb_classes, activation=lambda x: x,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out, name="head_add") / n_heads[-1]
    
        return logits

import torch
from torch import nn

class metrics_span(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, span_mask):
        '''
        logits.size()==(bsz,max_length,max_length,num_labels) .score for each span
        labels.size()==(bsz,max_length,max_length)
        span_mask.size()==(bsz,max_length,max_length)
        '''
        (bsz,max_length,max_length,num_labels)=logits.size()
        assert labels.size()==(bsz,max_length,max_length)==span_mask.size()
        #print(bsz,max_length,max_length,num_labels)

        span_mask.unsqueeze_(-1)#(bsz,max_length,max_length,1)
        assert span_mask.size()==(bsz,max_length,max_length,1)
        logits*=span_mask
        logits = torch.argmax(logits,dim=-1)#(bsz,max_length,max_length)
        assert logits.size()==(bsz,max_length,max_length)#label_id for each span

        logits=logits.view(size=(-1,)).float()
        labels=labels.view(size=(-1,)).float()

        ones=torch.ones_like(logits)
        zero=torch.zeros_like(logits)
        y_pred=torch.where(logits<1,zero,ones)
        y_pred=torch.triu(y_pred)#extract upper triangle matrix

        ones=torch.ones_like(labels)
        zero=torch.zeros_like(labels)
        y_true=torch.where(labels<1,zero,ones)#only golden span position is 1, otherwise positions are zeros

        corr=torch.eq(logits,labels).float()
        corr=torch.mul(corr,y_true)
        
        recall=torch.sum(corr)/(torch.sum(y_true)+1e-8)
        
        precision=torch.sum(corr)/(torch.sum(y_pred)+1e-8)
        
        f1=2*recall*precision/(recall+precision+1e-8)

        if torch.sum(labels)==0 and torch.sum(logits)==0:
            return 1.0,1.0,1.0

        return recall.item(), precision.item(), f1.item()


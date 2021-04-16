import training_params
import torch
import torch.nn as nn


def loss_function(output, target, mask, num_labels):
    cross_entropy_loss = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(cross_entropy_loss.ignore_index).type_as(target)
    )
    loss = cross_entropy_loss(active_logits, active_labels)
    return loss


class PunctuationModel(nn.Module):
    def __init__(self, num_tag):
        super(PunctuationModel, self).__init__()
        self.num_tag = num_tag
        self.bert = training_params.MODEL
        self.bert_drop = nn.Dropout(0.3)
        self.out_tag = nn.Linear(768, self.num_tag)

    def forward(self, ids, mask, token_type_ids, target_tag):
        o1, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)

        bo_tag = self.bert_drop(o1)

        tag = self.out_tag(bo_tag)

        loss_tag = loss_function(tag, target_tag, mask, self.num_tag)

        loss = loss_tag
        return tag, loss

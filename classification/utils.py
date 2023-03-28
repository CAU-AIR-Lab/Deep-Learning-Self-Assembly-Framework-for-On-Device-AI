import torch
import os
import torch.nn as nn
import pymysql

def get_module_list(max_priority, db_name, db_password):
    conn = pymysql.connect(host='localhost', user=db_name, password=db_password)
    cur = conn.cursor()
    cur.execute("USE nas_db")

    sql = "SELECT * FROM blocks ORDER BY priority ASC;"

    cur.execute(sql)
    result = cur.fetchall()
    module_list = []
    for row in result:
        row = row[1:]
        str_list = []
        str_list.append('_'.join([row[0], str(row[1])]))
        str_list.append('_'.join([row[2], row[3]]))
        str_list.append('_'.join([row[4], str(row[5])]))
        str_list.append(row[6])
        str_list.append(str(row[7])) 
        module_list.append('/'.join(str_list))
        if len(module_list) == max_priority:
            break
    cur.close()
    conn.close()
    return module_list

def check_tensor_in_list(atensor, alist):
    if any([(atensor == t_).all() for t_ in alist if atensor.shape == t_.shape]):
        return True
    return False
class AverageMeter(object):
    def __init__(self, name=''):
        self._name = name
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def reset(self):
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def __str__(self):
        return "%s: %.5f" % (self._name, self.avg)

    def get_avg(self):
        return self.avg

    def __repr__(self):
        return self.__str__()


def weights_init(m, deepth=0, max_depth=2):
    if deepth > max_depth:
        return
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, torch.nn.BatchNorm2d):
        return
    elif isinstance(m, torch.nn.ReLU):
        return
    elif isinstance(m, torch.nn.Module):
        deepth += 1
        for m_ in m.modules():
            weights_init(m_, deepth)
    else:
        raise ValueError("%s is unk" % m.__class__.__name__)




def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = torch.reshape(correct[:k], (-1,)).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res

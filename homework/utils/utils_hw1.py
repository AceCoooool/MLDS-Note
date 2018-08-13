import torch
from torch import autograd, nn
import scipy.linalg as LA


def eval_gradnorm(params):
    """
    eval gradient norm
    """
    grad_all = 0.0
    for p in params:
        grad = 0.0
        if p.grad is not None:
            grad = (p.grad.cpu().data.numpy() ** 2).sum()
        grad_all += grad
    return grad_all ** 0.5


# eval Hessian matrix
def eval_hessian(loss_grad, model):
    cnt = 0
    for g in loss_grad:
        g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
        cnt = 1
    l = g_vector.size(0)
    hessian = torch.zeros(l, l)
    for idx in range(l):
        grad2rd = autograd.grad(g_vector[idx], model.parameters())
        cnt = 0
        for g in grad2rd:
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        hessian[idx] = g2
    return hessian.cpu().data.numpy()


# eval minimal ratio
def minimal_ratio(loss_grad, model):
    hessian = eval_hessian(loss_grad, model)
    e = LA.eigvals(hessian)
    return (e > 1e-6).sum() / e.shape[0]


# eval trained model's loss and accuracy (for hw1_3_3)
def eval_model(model, dataloader, loss, metric, use_cuda=True):
    if use_cuda and torch.cuda.is_available():
        from torch.backends import cudnn
        cudnn.benchmark = True
        device = torch.device('cuda:0')
        model.to(device)
    total_loss, total_acc = 0, 0
    model.eval()
    with torch.no_grad():
        for idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += loss(output, target).item()
            total_acc += metric(output, target)
    avg_loss = total_loss / len(dataloader)
    avg_acc = 100 * total_acc / len(dataloader.dataset)
    return avg_loss, avg_acc


# eval sensitivity
def eval_sensitivity(model, dataloder, use_cuda=True):
    if use_cuda and torch.cuda.is_available():
        from torch.backends import cudnn
        cudnn.benchmark = True
        device = torch.device('cuda:0')
        model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(size_average=False)
    total_loss, total_sensitivity = 0, 0
    for idx, (data, target) in enumerate(dataloder):
        data = data.to(device).requires_grad_()
        target = target.to(device)
        output = model(data)
        loss_elem = criterion(output, target)
        total_loss += loss_elem.item()
        grad_x, = autograd.grad(loss_elem, data)
        for i in range(data.size(0)):
            total_sensitivity += torch.norm(grad_x[i]).item()
    avg_loss = total_loss / len(dataloder.dataset)
    avg_sensitivity = total_sensitivity / len(dataloder.dataset)
    return avg_loss, avg_sensitivity

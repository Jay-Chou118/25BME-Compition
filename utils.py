"""
作者：yueyue
日期：2023年10月30日
"""
CLASS = ['VF', 'SR', 'MA']
def adjust_lr(LR, epoch, optimzer):
    '''lr_schedule: 指数衰减，每个epoch变为前一个epoch的0.96倍'''
    lr = LR * (0.96 ** (epoch))
    print('lr', lr)
    for params_group in optimzer.param_groups:
        params_group['lr'] = lr
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def get_bcg_class(labels):
    '''获取标签对应的bcg类型'''
    text_labels = ['VF', 'SR', 'MA']
    return [text_labels[int(i)] for i in labels]
def accuracy2(y_hat, y, classes=CLASS):
    '''计算y_hat预测正确的数量'''
    correct_pred = {classname: 0 for classname in classes}
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    for label, prediction in zip(y, y_hat):
        if label == prediction:
            correct_pred[classes[label]] += 1

    return float(cmp.type(y.dtype).sum()), correct_pred

def accuracy(y_hat, y, classes=CLASS):
    '''计算y_hat预测正确的数量'''
    correct_pred = {classname: 0 for classname in classes}
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    for label, prediction in zip(y, y_hat):
        if label == prediction:
            correct_pred[classes[label]] += 1

    return  correct_pred

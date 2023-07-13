

def jaccard(labels, preds):
    #filter for unique values for all classes
    classes = set(list(labels))
    class_num = len(classes)
    
    if not (len(labels) == len(preds)):
        print("Labels and Preds are not equal long!")
        return -1
    
    '''
    TP - True Positive
    FP - False Positive
    FN - False Negative
    iou = \frac{ TP }{ TP + FP + FN} = \frac{INTERSECTION}{UNION}
    '''
    
    union = 0
    for i in range(len(labels)):
        if labels[i] == preds[i]:
            union+=1
    
    
    return 0

if __name__ == '__main__':
    labels = [1,2,3,1,2]
    preds = [1,2,3,1,1]
    
    iou = jaccard(labels, preds)
    print("IOU: {}".format(iou))
    
    
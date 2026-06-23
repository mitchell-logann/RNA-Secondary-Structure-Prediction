from sklearn.metrics import precision_score, recall_score, f1_score

def flattenContactMap(contact_map):
    return contact_map.flatten().cpu().numpy()

def evaluateContactMaps(true, pred):
    y_true = flattenContactMap(true)
    y_pred = flattenContactMap(pred)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred,zero_division=0)
    return precision, recall, f1
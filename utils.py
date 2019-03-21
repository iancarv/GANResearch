import pickle
import cv2
import numpy as np

def sliding_windows(img_size, window_size, step):
    max_x, max_y = img_size
    w, h = window_size
    x,y = 0,0
    windows = []
    for x_0 in range(x, max_x - w, step):
        for y_0 in range(y, max_y - h, step):
            windows.append((y_0,x_0,y_0+h,x_0+w))

    return np.array(windows)


def prepare_patches(patches):
    X_test = np.asarray(patches)
    X_test_resized = np.empty([X_test.shape[0], X_test.shape[3], 32, 32])
    for i in range(X_test.shape[0]):
        X_test_resized[i] = resize(X_test[i], (X_test.shape[3], 28, 28), mode='reflect')
    
    X_test_resized = 2 * X_test_resized - 1
    return X_test_resized


def is_nuclei(cell, t=17):      
    imgray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    p_c = (17, 17)
    for cnt in contours:
      dist = abs(cv2.pointPolygonTest(cnt,p_c,True))
      if dist < t:
        return True
    return False

def tp_fn(cell, windows, y_pred, y_scores, original):
  tp = 0
  fn = 0
  boxes = windows
  # grab the coordinates of the bounding boxes
  x1 = boxes[:,0]
  y1 = boxes[:,1]
  x2 = boxes[:,2]
  y2 = boxes[:,3]
  cenX = (x1 + x2) / 2
  cenY = (y1 + y2) / 2
  viz = original.copy()
  gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
  ret,gray = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
  im2, contours, hierarchy = cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#   cv2.drawContours(viz, contours, -1, (0,0,255), 5)
  selected = []
  not_found = []
  dSquared = 17*17
  for c in contours[1:]:
    M = cv2.moments(c)

    cY = int(M["m10"] / M["m00"])
    cX = int(M["m01"] / M["m00"])

#     in_area = ((x1 <= cX) & (cX <= x2)) & ((y1 <= cY) & (cY <= y2))
    dx = (cenX - cX)
    dy = (cenY - cY)
    
    in_area =  ((dx * dx) + (dy * dy) < dSquared)
    idxs = np.argwhere(in_area).reshape(-1)
#     [cv2.circle(viz,(window[0] + 17, window[1] + 17), 3, (0,255,0), -1) for window in windows[idxs]]
#     [cv2.circle(viz,(window[1] + 17, window[0] + 17), 3, (255,255,0), -1) for window in windows[np.argwhere(in_area & (y_pred == 0)).ravel()]]
#     print(y_pred[idxs].any())
    if y_pred[idxs].any():
      selected.append(idxs[np.argmax(y_scores[idxs])])
      tp += 1
      c = (0,255,0)
    else:
      fn += 1
      c = (255,0,0)
      if len(idxs):
        selected.append(idxs[np.argmax(y_scores[idxs])])
      else:
        not_found.append((cY, cX))
        c = (0,0,0)
    cv2.circle(viz,(cY, cX), 3, c, -1)
      
#     break
      
  return tp, fn, np.array(selected), np.array(not_found), viz

def test_model_metrics(gan, path, thresh_nms=0.3):
    print(path)
    print("Testing models metrics for thresh_nms:", str(thresh_nms))

    pickle_in = open(path,"rb")
    m = pickle.load(pickle_in)
    all_preds = None
    all_tests = None
    all_scores = None
    
    tps = []
    fps = []
    fns = []
    tns = []
    
    precision = []
    recall = []
    all_images = np.array(list(m.keys())[:1])

    results = {}
    for key in all_images:
        print(key)
        d = m[key]
        crop = d['crop']
        cell = d['cell']
        if crop.shape[0] != 400 or crop.shape[1] != 400:
          print('Menas')
          continue
        results[key] = {}
        windows = sliding_windows((400, 400), (34, 34), 6)
        patches = [crop[w[0]:w[2], w[1]:w[3]] for w in windows]
        cell_patches = [cell[w[0]:w[2], w[1]:w[3]] for w in windows]
        try:
            y_proba = sgan.predict_proba(prepare_patches(patches))
            y_proba = y_proba[1]
            y_pred = np.argmax(y_proba, axis=1)
            y_scores = y_proba[:,1]
        except Exception as e:
            print("Erro")
            print(e)
            continue
        else:
            pass
          
        y_test = np.array([is_nuclei(n) for n in cell_patches])
        results[key]['original'] = {
          'y_scores': y_scores,
          'y_test': y_test,
          'y_proba': y_proba,
          'y_pred': y_pred,
          'windows': windows,
        }
                
        fp = np.argwhere((y_pred == 1) & (y_test == 0)).ravel()
        tn = np.argwhere((y_pred == 0) & (y_test == 0)).ravel()
        
        tp, fn, arg_score, viz = tp_fn(cell, windows, y_pred, y_scores, crop)
        arg_score = np.concatenate([arg_score, fp, tn])
        y_scores = y_scores[arg_score]
        y_test = y_test[arg_score]
        y_pred = y_pred[arg_score]
        windows = windows[arg_score]
        y_proba = y_proba[arg_score]

        fp = len(fp)
        tps.append(tp)
        fns.append(fn)
        fps.append(fp)
        tns.append(tns)
        if tp == 0:
          precision.append(0)
          recall.append(0)
        else:
          prc = tp/(tp+fp)
          rec = tp/(tp+fn)
          precision.append(prc)
          recall.append(rec)
          
        results[key]['processed'] = {
          'y_scores': y_scores,
          'y_test': y_test,
          'y_proba': y_proba,
          'y_pred': y_pred,
          'picks': arg_score,
          'windows': windows,
        }
        if all_preds is None:
            all_preds = y_pred
            all_tests = y_test
            all_scores = y_scores
        else:
            all_preds = np.concatenate((all_preds, y_pred))
            all_tests = np.concatenate((all_tests, y_test))
            all_scores = np.concatenate((all_scores, y_scores))

    
    aveP = average_precision_score(all_tests, all_scores)
    avePred = average_precision_score(all_tests, all_preds)
    print ('\nOverall accuracy: %f%% \n' % (accuracy_score(all_tests, all_preds) * 100))
    print ('\nAveP: %f%% \n' % (aveP * 100))
    print ('\nAveP Preds: %f%% \n' % (avePred * 100))
    tps = np.array(tps)
    fns = np.array(fns)
    fps = np.array(fps)
    fps = np.array(fps)
    precision = np.array(precision)
    recall = np.array(recall)
    f1 = 2 * (precision * recall) / (precision + recall)
    idx = np.argsort(-precision)
    precision = precision[idx]
    idx = np.argsort(recall)
    recall = recall[idx]
    idx = np.argsort(f1)
    f1 = f1[idx]
    
    f_precision = np.sum(tps)/np.sum(tps+fps)
    f_recall = np.sum(tps)/np.sum(tps+fns)
    f_f1 = 2 * (f_precision * f_recall) / (f_precision + f_recall)
    print('Final Precision', f_precision)
    print('Final Recall', f_recall)
    print('Final F1', f_f1)
    
    # Calculating and ploting a Classification Report
    class_names = ['Non-nunclei', 'Nuclei']
    print('Classification report:\n %s\n'
        % (classification_report(all_tests, all_preds, target_names=class_names)))

    cm = confusion_matrix(all_tests, all_preds)
    print('Confusion matrix:\n%s' % cm)


    
    return aveP, avePred, all_tests, all_scores, all_preds, results

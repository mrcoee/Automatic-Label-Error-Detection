 
import numpy as np
cimport numpy as np
from copy import deepcopy

def segment_search( int i, int j, unsigned char[:,:] seg, short int seg_ind, np.ndarray marked_array, np.ndarray flag_array,
                    float[:,:,:] probs, unsigned char[:,:] gt, heatmaps, metrics,
                    unsigned short int[:,:] members_k, unsigned short int[:,:] members_l,
                    int nclasses, int x_max, int y_max ):
  
  cdef int k, l, ii, jj, x, y, n_in, n_bd, c, I, U, flag_max_x, flag_min_x, flag_max_y, flag_min_y, ic
  cdef unsigned char[:,:] flag
  cdef short int[:,:] marked
  
  if seg[i,j] <= 255: #<255
    
    n_in, n_bd = 0, 0
    c = seg[i,j]
    members_k[0,0], members_k[0,1] = i, j
    marked = marked_array
    
    flag_min_x = flag_max_x = i
    flag_min_y = flag_max_y = j
    
    flag = flag_array
    flag[i,j] = 1
    I, U = 0, 0
    marked[i,j] = seg_ind
    
    for m in metrics:
      metrics[m].append( 0 )
    
    # go through union of current segment and corresponding ground truth
    # and identify all inner pixels, boundary pixels and
    # pixels where ground_truth and prediction match
    k = 1
    l = 0
    num_neighbors = 0      
    while k > 0 or l > 0:
      
      flag_k = 0
      
      if k > 0:
        k -= 1
        x, y = members_k[k]
        flag_k = 1
      elif l > 0:
        l -= 1
        x, y = members_l[l]

      if flag_k:
      
        for ii in range(max(x-1,0),min(x+2,x_max)):
          for jj in range(max(y-1,0),min(y+2,y_max)):
            if seg[ii,jj] == c and marked[ii,jj] == 0:
              marked[ii,jj] = seg_ind
              flag[ii,jj] = 1
              if ii > flag_max_x:
                flag_max_x = ii
              elif ii < flag_min_x:
                flag_min_x = ii
              if jj > flag_max_y:
                flag_max_y = jj
              elif jj < flag_min_y:
                flag_min_y = jj
              members_k[k,0], members_k[k,1] = ii, jj
              k += 1
            elif seg[ii,jj] != c:
              if seg[ii,jj] <= 255: #!=
                metrics["ndist"+str(seg[ii,jj])][-1] = metrics["ndist"+str(seg[ii,jj])][-1]+1
              num_neighbors += 1
              marked[x,y] = -seg_ind
              if gt != []:
                if gt[ii,jj] == c and flag[ii,jj]==0:
                  flag[ii,jj] = 1
                  if ii > flag_max_x:
                    flag_max_x = ii
                  elif ii < flag_min_x:
                    flag_min_x = ii
                  if jj > flag_max_y:
                    flag_max_y = jj
                  elif jj < flag_min_y:
                    flag_min_y = jj
                  members_l[l,0], members_l[l,1] = ii, jj
                  l += 1

      if not flag_k and gt != []:
        if I == 0:
          break
        for ii in range(max(x-1,0),min(x+2,x_max)):
          for jj in range(max(y-1,0),min(y+2,y_max)):
            #if gt[ii,jj] == c and flag[ii,jj]==0: # ordinary IoU
            if gt[ii,jj] == c and flag[ii,jj]==0 and seg[ii,jj] != c: # IoU_adj
              flag[ii,jj] = 1
              if ii > flag_max_x:
                flag_max_x = ii
              elif ii < flag_min_x:
                flag_min_x = ii
              if jj > flag_max_y:
                flag_max_y = jj
              elif jj < flag_min_y:
                flag_min_y = jj
              members_l[l,0], members_l[l,1] = ii, jj
              l += 1

      if flag_k:
        if marked[x,y] in [seg_ind,-seg_ind]:
          # update heap maps
          if marked[x,y] == seg_ind:
            for h in heatmaps:
              metrics[h+"_in"][-1] += heatmaps[h][x,y]
              metrics[h+"_var_in"][-1] += heatmaps[h][x,y]**2
            n_in += 1
          elif marked[x,y] == -seg_ind:
            for h in heatmaps:
              metrics[h+"_bd"][-1] += heatmaps[h][x,y]
              metrics[h+"_var_bd"][-1] += heatmaps[h][x,y]**2
            n_bd += 1
          for ic in range(nclasses):
            metrics["cprob"+str(ic)][-1] += probs[x,y,ic]
          metrics["mean_x"][-1] += x
          metrics["mean_y"][-1] += y
          if gt != []:
            if gt[x,y] == c:
              I += 1
              
      U += 1
      
    for ii in range(flag_min_x,flag_max_x+1):
      for jj in range(flag_min_y,flag_max_y+1):
        flag[ii,jj] = 0
    
    # compute all metrics
    metrics["class"   ][-1] = c
    if gt != []:
      metrics["iou"     ][-1] = float(I) / float(U)
      metrics["iou0"    ][-1] = int(I == 0)
    else:
      metrics["iou"     ][-1] = -1
      metrics["iou0"    ][-1] = -1
    metrics["S"       ][-1] = n_in + n_bd
    metrics["S_in"    ][-1] = n_in
    metrics["S_bd"    ][-1] = n_bd
    metrics["S_rel"   ][-1] = float( n_in + n_bd ) / float(n_bd)
    metrics["S_rel_in"][-1] = float( n_in ) / float(n_bd)
    metrics["mean_x"][-1] /= ( n_in + n_bd )
    metrics["mean_y"][-1] /= ( n_in + n_bd )
    
    for nc in range(nclasses):
      metrics["cprob"+str(nc)][-1] /= ( n_in + n_bd )
    
    for nc in range(nclasses):
      metrics["ndist"+str(nc)][-1] /= float(np.max((num_neighbors,1)))
    
    for h in heatmaps:
      metrics[h          ][-1] = (metrics[h+"_in"][-1] + metrics[h+"_bd"][-1]) / float( n_in + n_bd )
      
      if ( n_in > 0 ):
        metrics[      h+"_in"][-1] /= float(n_in)
      metrics[h+"_bd"    ][-1] /= float(n_bd)
      
      metrics[h+"_var"   ][-1] = (metrics[h+"_var_in"][-1] + metrics[h+"_var_bd"][-1]) / float( n_in + n_bd ) - (metrics[h][-1] **2 )
      
      if ( n_in > 0 ):
        metrics[h+"_var_in"][-1] = metrics[h+"_var_in"][-1] / float(n_in) - metrics[h+"_in"][-1]**2
      metrics[h+"_var_bd"][-1] = metrics[h+"_var_bd"][-1] / float(n_bd) - metrics[h+"_bd"][-1]**2  
        
      metrics[h+"_rel"   ][-1] = metrics[h      ][-1] * metrics["S_rel"   ][-1]
      metrics[h+"_rel_in"][-1] = metrics[h+"_in"][-1] * metrics["S_rel_in"][-1]
      
      metrics[h+"_var_rel"   ][-1] = metrics[h+"_var"   ][-1] * metrics["S_rel"   ][-1]
      metrics[h+"_var_rel_in"][-1] = metrics[h+"_var_in"][-1] * metrics["S_rel_in"][-1]


    seg_ind +=1
      
  return marked_array, metrics, seg_ind 



def entropy( probs ):

  E = np.sum( np.multiply( probs, np.log(probs+np.finfo(np.float32).eps) ) , axis=-1) / np.log(1.0/probs.shape[-1])
  return np.asarray( E, dtype="float32" )



def probdist( probs ):

  cdef int i, j
  
  arrayA = np.asarray(np.argsort(probs,axis=-1), dtype="uint8")
  arrayD = np.ones( probs.shape[:-1], dtype="float32" )
  
  cdef float[:,:,:] P = probs
  cdef float[:,:]   D = arrayD
  cdef char[:,:,:]  A = arrayA
  
  for i in range( arrayD.shape[0] ):
    for j in range( arrayD.shape[1] ):
      D[i,j] = ( 1 - P[ i, j, A[i,j,-1] ] + P[ i, j, A[i,j,-2] ] )
  
  return arrayD


def varrat( probs ):

  cdef int i, j
  
  arrayA = np.asarray(np.argsort(probs,axis=-1), dtype="uint8")
  arrayV = np.ones( probs.shape[:-1], dtype="float32" )
  
  cdef float[:,:,:] P = probs
  cdef float[:,:]   V = arrayV
  cdef char[:,:,:]  A = arrayA
  
  for i in range( arrayV.shape[0] ):
    for j in range( arrayV.shape[1] ):
      V[i,j] = ( 1 - P[ i, j, A[i,j,-1] ] )
  
  return arrayV



def prediction(probs, gt, ignore=True ):
    
  pred = np.asarray( np.argmax( probs, axis=-1 ), dtype="uint8" )

  if ignore == True:
    pred[ gt==255 ] = 255
  
  return pred



def compute_metrics_components(probs, gt):
  
  cdef int i, j
  cdef short int seg_ind
  cdef np.ndarray marked
  cdef np.ndarray tmarked
  cdef np.ndarray members_k
  cdef np.ndarray members_l
  cdef short int[:,:] M
  cdef short int[:,:] tM
  
  nclasses  = probs.shape[-1]
  dims      = np.asarray( probs.shape[:-1], dtype="uint16" )
  gt        = np.asarray( gt, dtype="uint8" )
  probs     = np.asarray( probs, dtype="float32" )
  seg       = np.asarray( prediction(probs, gt, ignore=False ), dtype="uint8" )
  marked    = np.zeros( dims, dtype="int16" )
  tmarked   = np.zeros( dims, dtype="int16" )
  members_k = np.zeros( (np.prod(dims), 2 ), dtype="uint16" )
  members_l = np.zeros( (np.prod(dims), 2 ), dtype="uint16" )
  flag      = np.zeros( dims, dtype="uint8" )
  M         = marked
  tM        = tmarked
  
  heatmaps = { "E": entropy( probs ), "D": probdist( probs ), "V": varrat( probs ) }
  
  metrics = { "iou": list([]), "iou0": list([]), "class": list([]), "mean_x": list([]), "mean_y": list([]) }
  
  for m in list(heatmaps)+["S"]:
    metrics[m          ] = list([])
    metrics[m+"_in"    ] = list([])
    metrics[m+"_bd"    ] = list([])
    metrics[m+"_rel"   ] = list([])
    metrics[m+"_rel_in"] = list([])
    if m != "S":
      metrics[m+"_var"   ] = list([])
      metrics[m+"_var_in"    ] = list([])
      metrics[m+"_var_bd"    ] = list([])
      metrics[m+"_var_rel"   ] = list([])
      metrics[m+"_var_rel_in"] = list([])
    
    
  for i in range(nclasses):
    metrics['cprob'+str(i)] = list([])
    
  for i in range(nclasses):
    metrics['ndist'+str(i)] = list([])

  t_metrics = deepcopy(metrics)
  
  seg_ind = 1
  tseg_ind = 1

  for i in range(dims[0]):
    for j in range(dims[1]):
      if M[i,j] == 0:
        
        marked, metrics, seg_ind = segment_search( i, j, seg, seg_ind, marked, flag, probs, gt, heatmaps, metrics, members_k, members_l, nclasses, dims[0], dims[1] )
  
  return metrics, marked


def compute_IoU( int i, int j, unsigned char[:,:] seg, short int seg_ind, np.ndarray marked_array, np.ndarray flag_array,
                    unsigned char[:,:] gt, metrics, unsigned short int[:,:] members_k, unsigned short int[:,:] members_l, 
                    int x_max, int y_max ):
  
  #marked: map that for each image pixel contains the id (seg_ind) of the predicted segment 
  #flag: helper array that registers whether pixel ii, jj has already been visited 
  #seg: (predicted) segmentation mask of which we compute the connected components
  #gt: ground truth (or reference) segmentation mask  
  #x_max, y_max: image size
  #flag_max_x, flag_min_x, flag_max_y, flag_min_y: describe the range of the part of the (predicted) segment which has already been considered
  #i, j: start pixel of the (predicted) segment
  #x, y: pixel of (predicted) segment currently being inspected
  #ii, jj: indices for inspecting the neighborhood of x, y
  #members_k: list of pixel coordinates being inspected during breadth-first search over the (predicted) segment ...
  #k: help index for list access to members_k
  #flag_k: 0 if the predicted segment has been entirely inspected, 1 else 
  #members_l: list of pixel coordinates beeing inspected during breadth-first search over all ground truth segments that intersect with the predicted segment
  #l: help index for list access to members_l


  cdef int k, l, ii, jj, x, y, c, I, U, flag_max_x, flag_min_x, flag_max_y, flag_min_y
  cdef unsigned char[:,:] flag
  cdef short int[:,:] marked
  
  if seg[i,j] <= 255: #<
    
    c = seg[i,j]
    members_k[0,0], members_k[0,1] = i, j
    marked = marked_array
    
    flag_min_x = flag_max_x = i
    flag_min_y = flag_max_y = j
    
    flag = flag_array
    flag[i,j] = 1
    I, U = 0, 0
    marked[i,j] = seg_ind
    
    for m in metrics:
      metrics[m].append( 0 )
    
    # go through union of current segment and corresponding ground truth
    # and identify all inner pixels, boundary pixels and
    # pixels where ground_truth and prediction match
    k = 1
    l = 0
    num_neighbors = 0      
    while k > 0 or l > 0:
      #while there are still pixels of the predicted segment left (k>0)
      #or of the overlapping ground truth segments (l>0)

      flag_k = 0
      
      if k > 0:
        k -= 1
        x, y = members_k[k]
        flag_k = 1
      elif l > 0:
        l -= 1
        x, y = members_l[l]
      
      #considers each of the eight neighboring pixels ii, jj of x, y and check if they belong to the segment. If so, add ii, jj to list members_k  
      if flag_k:
        for ii in range(max(x-1,0),min(x+2,x_max)):
          for jj in range(max(y-1,0),min(y+2,y_max)):
            if seg[ii,jj] == c and marked[ii,jj] == 0:
                marked[ii,jj] = seg_ind
                flag[ii,jj] = 1
                if ii > flag_max_x:
                  flag_max_x = ii
                elif ii < flag_min_x:
                  flag_min_x = ii
                if jj > flag_max_y:
                  flag_max_y = jj
                elif jj < flag_min_y:
                  flag_min_y = jj
                members_k[k,0], members_k[k,1] = ii, jj
                k += 1
            elif seg[ii,jj] != c:
                num_neighbors += 1
                marked[x,y] = -seg_ind #mark as boundary pixel of the segment
                if gt != []:
                  if gt[ii,jj] == c and flag[ii,jj]==0:
                    flag[ii,jj] = 1
                    if ii > flag_max_x:
                      flag_max_x = ii
                    elif ii < flag_min_x:
                      flag_min_x = ii
                    if jj > flag_max_y:
                      flag_max_y = jj
                    elif jj < flag_min_y:
                      flag_min_y = jj
                    members_l[l,0], members_l[l,1] = ii, jj
                    l += 1

      if not flag_k and gt != []:
        #if the predicted segment has been entirely inspected

        if I == 0:
          #if intesection is empty, stop (IoU=0)
          break
        for ii in range(max(x-1,0),min(x+2,x_max)):
          for jj in range(max(y-1,0),min(y+2,y_max)):
            #if gt[ii,jj] == c and flag[ii,jj]==0: # ordinary IoU
            if gt[ii,jj] == c and flag[ii,jj]==0 and seg[ii,jj] != c: # IoU_adj
              flag[ii,jj] = 1
              if ii > flag_max_x:
                flag_max_x = ii
              elif ii < flag_min_x:
                flag_min_x = ii
              if jj > flag_max_y:
                flag_max_y = jj
              elif jj < flag_min_y:
                flag_min_y = jj
              members_l[l,0], members_l[l,1] = ii, jj
              l += 1

      if flag_k:
        if marked[x,y] in [seg_ind,-seg_ind]:
          #if pixel in predicted segment

          if gt != []:
            if gt[x,y] == c:
              #if corresponding ground truth pixel is of the same class
              I += 1
              
      U += 1
    
    #flag array must be set to zero for later reuses (for subsequent segments)
    for ii in range(flag_min_x,flag_max_x+1):
      for jj in range(flag_min_y,flag_max_y+1):
        flag[ii,jj] = 0
    
    # compute IoU
    if gt != []:
      metrics["iou"][-1] = float(I) / float(U)
      metrics["iou0"][-1] = int(I == 0)
    else:
      metrics["iou"][-1] = -1
      metrics["iou0"][-1] = -1

    seg_ind +=1
      
  return marked_array, metrics, seg_ind


def compute_IoU_component(pred, gt, is_prob=False):

  cdef int i, j
  cdef short int seg_ind
  cdef np.ndarray marked
  cdef np.ndarray members_k
  cdef np.ndarray members_l
  cdef short int[:,:] M
  cdef float IoU

  gt   = np.asarray( gt, dtype="uint8" )
  if is_prob:
    seg = np.asarray( prediction(pred, gt, ignore=False ), dtype="uint8" )
    dims = np.asarray( pred.shape[:-1], dtype="uint16" )
  else:
    seg = np.asarray( pred, dtype="uint8" )
    dims = np.asarray( pred.shape, dtype="uint16" )

  marked    = np.zeros( dims, dtype="int16" )
  members_k = np.zeros( (np.prod(dims), 2 ), dtype="uint16" )
  members_l = np.zeros( (np.prod(dims), 2 ), dtype="uint16" )
  flag      = np.zeros( dims, dtype="uint8" )
  M         = marked
  
  seg_ind = 1

  IoUs = { "iou": list([]), "iou0": list([]) }
  
  for i in range(dims[0]):
    for j in range(dims[1]):
      if M[i,j] == 0:
        
        marked, IoUs, seg_ind = compute_IoU( i, j, seg, seg_ind, marked, flag, gt,  IoUs, members_k, members_l , dims[0], dims[1])
  
  return IoUs
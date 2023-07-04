import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
import h5py
import time
import random
from itertools import chain, islice
from queue import Queue
from threading import Thread



class ParallelLoader:
    
    def __init__(self, datasets, buffer_size, weak_labels=False, expand_weak_labels=True):
        self.datasets=datasets
        self.data_queue = Queue(buffer_size)
        self.sentinel = object() # marker that one epoch is over and the iterator can end
        self.weak_labels = weak_labels
        self.expand_weak_labels = expand_weak_labels
        
    def get_stream_loaders(self):
        return zip(*[DataLoader(dataset, num_workers=1, batch_size=None)
                   for dataset in self.datasets])

    def fill_queue(self):
        for batch_parts in self.get_stream_loaders():
            dat, lab = list(zip(*list(chain(*batch_parts))))
            # shape differently sized target vectors to same size by repeating the last target vector
            if self.weak_labels and not self.expand_weak_labels:
                # get longest target vector
                max_len = max([element.shape[1] for element in lab])
                new_lab = []
                for element in lab:
                    new_element = torch.zeros((element.shape[0], max_len, element.shape[2]))
                    new_element[:,0:element.shape[1],:] = element
                    new_element[:,element.shape[1]:,:] = element[:,-1,:] # repeat the last target vector
                    new_lab.append(new_element)
                lab = new_lab
            self.data_queue.put((torch.stack(dat), torch.stack(lab)))
        # epoch ended
        self.data_queue.put(self.sentinel)
        self.data_queue.join()
    

    def __iter__(self):
        worker_thread = Thread(target=self.fill_queue)
        worker_thread.start()
        while True:
            retval = self.data_queue.get()
            if retval == self.sentinel:
                self.data_queue.task_done()
                break
            self.data_queue.task_done()
            yield retval
        worker_thread.join()



class ParallelHCQTLoader(ParallelLoader):
    def __init__(self, fileList=None, cqtList=None, labelsList=None, segments_per_file=None, avg_frames_per_segment=None, target_indices=None, stride=None,
                 batch_size=32, numContextFrames=75, compression=10, verbose=False, shuffle=False, jointContextLoading=False,
                 data_label="hcqt", target_label="chroma", max_workers=1, cycle=False, cache_file=False, num_targets=1, weak_labels=False, expand_weak_labels=True, buffer_size=5,
                return_nReps=False):
        
        self.datasets = HCQTdataset_h5.split_datasets(fileList=fileList, 
                    segments_per_file=segments_per_file, 
                    max_workers=max_workers,
                    avg_frames_per_segment=avg_frames_per_segment, 
                    target_indices=target_indices,
                    stride=stride,
                    batch_size=batch_size, 
                    numContextFrames=numContextFrames,
                    compression=compression,
                    verbose=verbose,
                    shuffle=shuffle,
                    jointContextLoading=jointContextLoading,
                    data_label=data_label,
                    target_label=target_label,
                    cycle=cycle,
                    cache_file=cache_file,
                    num_targets=num_targets,
                    weak_labels=weak_labels,
                    expand_weak_labels=expand_weak_labels,
                    return_nReps=return_nReps)
        
        super().__init__(self.datasets, buffer_size, weak_labels, expand_weak_labels)
            
            
class HCQTdataset_h5(IterableDataset):
    def __init__(self, fileList=None, cqtList=None, labelsList=None, segments_per_file=None, avg_frames_per_segment=None, target_indices=None, stride=None,
                 batch_size=32, numContextFrames=75, compression=10, verbose=False, shuffle=False, jointContextLoading=False,
                 data_label="hcqt", target_label="chroma", cycle=False, cache_file=False, num_targets=1, weak_labels=False, expand_weak_labels=True, return_nReps=False):
        self.verbose=verbose
        if self.verbose:
            print("init")
        
        
        assert (np.abs((segments_per_file is None) + (avg_frames_per_segment is None) + (target_indices is None) +(stride is None) - 3) <1e-3),\
        "exactly one argument of segments_per_file, avg_frames_per_segment, target_indices, or stride must be specified"
        
        self.oneSideContext = int((numContextFrames-num_targets)/2)        
        self.fileList = fileList                
        self.batch_size = batch_size
        self.compression = compression
        self.numContextFrames = numContextFrames
        self.segments_per_file=segments_per_file
        self.avg_frames_per_segment=avg_frames_per_segment
        self.target_indices=target_indices        
        self.stride=stride
        self.shuffle=shuffle        
        self.jointContextLoading=jointContextLoading        
        self.data_label=data_label
        self.target_label=target_label
        self.cycle=cycle
        self.cache_file=cache_file
        self.num_targets=num_targets
        self.weak_labels=weak_labels
        self.expand_weak_labels=expand_weak_labels
        
        self.return_nReps=return_nReps
    
    # return pairs of data & target for file
    def process_data(self, file):
        if self.verbose:
            worker = torch.utils.data.get_worker_info()
            worker_id = id(self) if worker is not None else -1
            print("worker %i, load %s"%(worker_id%1000, file.split("/")[-1]))
        else:
            worker_id=-2
            
        hf_read = h5py.File(file, "r")
        
        reader = {"data": hf_read[self.data_label],
                  "target": hf_read[self.target_label]}
        
        if self.cache_file:
            #print("cache file")
            reader["data"] = torch.Tensor(reader["data"][:])
            reader["target"] = torch.Tensor(reader["target"][:])
        
        fileFrames = reader["data"].shape[1]
        
        if self.segments_per_file is not None:
            centerInds = list(np.random.randint(self.oneSideContext+1,fileFrames-self.oneSideContext-self.num_targets,self.segments_per_file))
        elif self.avg_frames_per_segment is not None:
            centerInds = list(np.random.randint(self.oneSideContext+1,fileFrames-self.oneSideContext-self.num_targets,int(fileFrames / self.avg_frames_per_segment)))
        elif self.target_indices is not None:
            centerInds = self.target_indices
        elif self.stride is not None:
            centerInds = list(np.arange(self.oneSideContext+1, fileFrames-self.oneSideContext-self.num_targets, self.stride))
        else:
            raise Exception("no indices selection method specified")
        
        if self.verbose:
            print("loaded center inds:")
            print(centerInds)        
        
        # load frames only once; THIS IS SLOW!!!
        # not supported any more to reduce the effort of changing code
        if self.jointContextLoading:
            raise Exception("not supported")

        # don't care if the same frames are loaded multiple times    
        else:
            for ind in centerInds:
                if self.verbose:
                    print("worker %i, yield index %i"%(worker_id%1000, ind))
                    
                data = reader["data"][:,ind-self.oneSideContext:ind+self.oneSideContext+self.num_targets,:]
                if self.compression is not None:
                    data = np.log(1+self.compression*data)

                target = reader["target"][...,ind:ind+self.num_targets]
                target = np.swapaxes(target, 1, 3)
                target = np.squeeze(target, 3)
                
                if self.weak_labels:
                    new_target = None
                    last_target = None
                    
                    for frame in range(target.shape[1]):
                        # first frame
                        if frame == 0:
                            new_target = target[:,frame,:]
                            last_target = target[:,frame,:]
                            continue   
                        # check if new label
                        if not np.array_equal(target[:,frame,:], last_target):
                            last_target = target[:,frame,:]
                            new_target = np.concatenate((new_target,last_target),0)
                    
                    if self.expand_weak_labels:
                        unfold_indices = np.linspace(0, new_target.shape[0], target.shape[1], endpoint=False).astype(int)
                        target[:,:,:] = new_target[unfold_indices,:]
                        
                    else:
                        target = np.expand_dims(new_target, 0)

                yield data, target
            hf_read.close()
        
    # cycle endlessly through iterator (it), shuffle if desired
    def cycleShuffle(self, it):
        saved = []
        for el in it:
            yield el
            saved.append(el)
        
        
        if self.cycle:
            while saved:
                if self.shuffle:
                    random.shuffle(saved)
                    print("shuffle list")
                for el in saved:
                    yield el
        
    # get an endless stream from process_data
    def get_stream(self):
        if self.verbose:
            worker = torch.utils.data.get_worker_info()
            worker_id = id(self) if worker is not None else -1
            print("worker %i get_stream()"%(worker_id%1000))
        return chain.from_iterable(map(self.process_data, self.cycleShuffle(self.fileList)))
    
    # buffer stream until (splitted) batch size is reached, then yield batch
    def get_batched_stream(self):
        while batch := list(islice(iter(self.get_stream()), self.batch_size)):   
            if self.verbose:
                worker = torch.utils.data.get_worker_info()
                worker_id = id(self) if worker is not None else -1
                print("worker %i yield batch"%(worker_id%1000))
            yield batch
    
    def batched(self, iterable, n):
        "Batch data into lists of length n. The last batch may be shorter."
        # batched('ABCDEFG', 3) --> ABC DEF G
        it = iter(iterable)
        while True:
            batch = list(islice(it, n))
            if not batch:
                return
            yield batch

    # iterator
    def __iter__(self):
        return self.batched(self.get_stream(), self.batch_size)

    
            
    @classmethod
    def split_datasets(cls, fileList, segments_per_file=None, avg_frames_per_segment=None, target_indices=None, stride=None,
                 batch_size=32, numContextFrames=75, compression=10, shuffle=True, max_workers=1, verbose=False,
                 jointContextLoading=False, data_label="hcqt", target_label="chroma", cycle=False, cache_file=False,
                 num_targets=1, weak_labels=False, expand_weak_labels=True, return_nReps=False):
        
        # make sure that batch size is divisible by num_workers
        for n in range(max_workers, 0, -1):
            if batch_size % n == 0:
                num_workers=n
                break
        split_size = batch_size // num_workers
        
        if shuffle:
            random.shuffle(fileList)

        
        return [cls(fileList=fileList[i::num_workers], 
                    segments_per_file=segments_per_file, 
                    avg_frames_per_segment=avg_frames_per_segment, 
                    target_indices=target_indices,
                    stride=stride,
                    batch_size=split_size, 
                    numContextFrames=numContextFrames,
                    compression=compression,
                    verbose=verbose,
                    shuffle=shuffle,
                    jointContextLoading=jointContextLoading,
                    data_label=data_label,
                    target_label=target_label,
                    cycle=cycle,
                    cache_file=cache_file,
                    num_targets=num_targets,
                    weak_labels=weak_labels,
                    expand_weak_labels=expand_weak_labels,
                    return_nReps=return_nReps)
                for i in range(num_workers)]        
            
            

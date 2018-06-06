import tensorflow as tf
import numpy as np
import os
from utils import eprint, listdir_files, reset_random, create_session, BatchPNG
from input import inputs, input_arguments

class Save:
    def __init__(self, config):
        self.dataset = None
        self.save_dir = None
        self.training = None
        self.num_epochs = None
        self.max_steps = None
        self.random_seed = None
        self.batch_size = None
        # copy all the properties from config object
        self.config = config
        self.__dict__.update(config.__dict__)
    
    def initialize(self):
        # create save directory
        if os.path.exists(self.save_dir):
            eprint('Confirm removing {}\n[Y/n]'.format(self.save_dir))
            if input() != 'Y':
                exit
            import shutil
            shutil.rmtree(self.save_dir)
            eprint('Removed: ' + self.save_dir)
        os.makedirs(self.save_dir)
        # set deterministic random seed
        if self.random_seed is not None:
            reset_random(self.random_seed)
    
    def get_dataset(self):
        files = listdir_files(self.dataset, filter_ext=['.jpeg', '.jpg', '.png'])
        # random shuffle
        import random
        random.shuffle(files)
        # size of dataset
        self.epoch_steps = len(files) // self.batch_size
        self.epoch_size = self.epoch_steps * self.batch_size
        if not self.training:
            self.num_epochs = 1
            self.max_steps = self.epoch_steps
        if self.max_steps is None:
            self.max_steps = self.epoch_steps * self.num_epochs
        else:
            self.num_epochs = (self.max_steps + self.epoch_steps - 1) // self.epoch_steps
            self.config.num_epochs = self.num_epochs
        self.files = files[:self.epoch_size]
        eprint('data set: {}\nepoch steps: {}\nnum epochs: {}\nmax steps: {}\n'
            .format(len(self.files), self.epoch_steps, self.num_epochs, self.max_steps))
    
    def build_input(self):
        with tf.device('/cpu:0'):
            self.inputs, self.labels = inputs(
                self.config, self.files, is_training=self.training)
    
    def save(self, sess):
        epoch_len = len(str(self.num_epochs - 1))
        step_len = len(str(self.epoch_steps - 1))
        for epoch in range(self.num_epochs):
            save_dir = os.path.join(self.save_dir, '{:0>{epoch_len}}'
                .format(epoch, epoch_len=epoch_len))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            num_steps = min(self.epoch_steps, self.max_steps - self.epoch_steps * epoch)
            for step in range(num_steps):
                ret_inputs, ret_labels = sess.run((self.inputs, self.labels))
                if self.batch_size == 1:
                    ret_inputs = ret_inputs[0]
                    ret_labels = ret_labels[0]
                # save to compressed npz file
                ofile = os.path.join(save_dir, '{:0>{step_len}}'
                    .format(step, step_len=step_len))
                np.savez_compressed(ofile, inputs=ret_inputs, labels=ret_labels)
    
    def __call__(self):
        self.initialize()
        self.get_dataset()
        with tf.Graph().as_default():
            self.build_input()
            with create_session() as sess:
                self.save(sess)

def main(argv=None):
    # arguments parsing
    import argparse
    argp = argparse.ArgumentParser()
    # testing parameters
    argp.add_argument('dataset')
    argp.add_argument('save_dir')
    argp.add_argument('--training', action='store_true')
    argp.add_argument('--num-epochs', type=int, default=32)
    argp.add_argument('--max-steps', type=int)
    argp.add_argument('--random-seed', type=int)
    argp.add_argument('--batch-size', type=int, default=1)
    # data parameters
    argp.add_argument('--dtype', type=int, default=2)
    argp.add_argument('--data-format', default='NCHW')
    argp.add_argument('--patch-height', type=int, default=128)
    argp.add_argument('--patch-width', type=int, default=128)
    argp.add_argument('--in-channels', type=int, default=3)
    argp.add_argument('--out-channels', type=int, default=3)
    # pre-processing parameters
    input_arguments(argp)
    # model parameters
    argp.add_argument('--scaling', type=int, default=1)
    # parse
    args = argp.parse_args(argv)
    args.dtype = [tf.int8, tf.float16, tf.float32, tf.float64][args.dtype]
    # run saving
    save = Save(args)
    save()

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])

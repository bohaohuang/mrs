# Built-in
import os

# Libs

# Pytorch

# Own modules
from mrs_utils import misc_utils


class BasicProcess(object):
    """
    Process block is a basic running module for this repo, it will run the process by checking if function has been
    ran before, or be forced to re-run the process again
    """
    def __init__(self, name, path, func=None):
        """
        :param name:name of the process, this will be used for the state file name
        :param path: path to where the state file will be stored
        :param func: process function, if None then it will be child class's process() function
        """
        self.name = name
        self.path = path
        if func is None:
            self.func = self.process
        else:
            self.func = func
        self.state_file = os.path.join(self.path, '{}_state.txt'.format(self.name))

    def process(self, **kwargs):
        raise NotImplementedError()

    def run(self, force_run=False, **kwargs):
        """
        Run the process
        :param force_run: if True, then the process will run no matter it has completed before
        :param kwargs:
        :return:
        """
        # check if state file exists
        state_exist = os.path.exists(self.state_file)
        # run the function if force run or haven't run before
        if force_run or state_exist == 0:
            print(('Start running {}'.format(self.name)))
            # write state log as incomplete
            with open(self.state_file, 'w') as f:
                f.write('Incomplete\n')

            # run the process
            self.func(**kwargs)

            # write state log as complete
            with open(self.state_file, 'w') as f:
                f.write('Finished\n')
        else:
            # if haven't run before, run the process
            if not self.check_finish():
                self.func(**kwargs)

            # write state log as complete
            with open(self.state_file, 'w') as f:
                f.write('Finished\n')
        return self

    def check_finish(self):
        """
        check if state file exists
        :return: True if it has finished
        """
        state_exist = os.path.exists(self.state_file)
        if state_exist:
            with open(self.state_file, 'r') as f:
                a = f.readlines()
                if a[0].strip() == 'Finished':
                    return True
        return False


class ValueComputeProcess(BasicProcess):
    """
    Compute value for the given function, save value
    Return the value if already exists
    """
    def __init__(self, name, path, save_path, func=None):
        """
        :param name:name of the process, this will be used for the state file name
        :param path: path to where the state file will be stored
        :param save_path: path to save the computed value
        :param func: process function, if None then it will be child class's process() function
        """
        self.save_path = save_path
        self.val = None
        super().__init__(name, path, func)

    def run(self, force_run=False, **kwargs):
        """
        Run the process
        :param force_run: if True, then the process will run no matter it has completed before
        :param kwargs:
        :return:
        """
        # check if state file exists
        state_exist = os.path.exists(self.state_file)
        # run the function if force run or haven't run before
        if force_run or state_exist == 0:
            print(('Start running {}'.format(self.name)))
            # write state log as incomplete
            with open(self.state_file, 'w') as f:
                f.write('Incomplete\n')

            # run the process
            self.val = self.func(**kwargs)

            # write state log as complete
            with open(self.state_file, 'w') as f:
                f.write('Finished\n')
            misc_utils.save_file(self.save_path, self.val)
        else:
            # if haven't run before, run the process
            if not self.check_finish():
                self.val = self.func(**kwargs)
                misc_utils.save_file(self.save_path, self.val)

            # if already exists, load the file
            self.val = misc_utils.load_file(self.save_path)

            # write state log as complete
            with open(self.state_file, 'w') as f:
                f.write('Finished\n')
        return self
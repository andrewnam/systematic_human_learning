from multiprocessing import Process, Semaphore, Manager
from tqdm.auto import tqdm


def run_func_gpu(func, device, semaphore, return_dict, **kwargs):
    """
    A wrapper used by GPUMultiprocessor
    """
    result = func(**kwargs, device=device)
    return_dict['result'] = result
    return_dict['device'] = device
    semaphore.release()


class GPUMultiprocessor:
    """
    A class for running embarrassingly parallel code across multiple GPUs.
    Queues up processes so that as soon as one process finishes, the next enqueued process takes the GPU.
    To avoid memory issues, only one process is assigned to each GPU regardless of how much memory it takes.
    """

    def __init__(self, func, df_kwargs, devices, show_pbar=True):
        """
        :param func: function to run in parallel, must take 'device' and every column of df_kwargs as its arguments
        :param df_kwargs: a DataFrame containing the arguments to run in the function
            A new process is enqueued for each row in the DataFrame
            Order is preserved in the returned list
        :param devices: a list of GPUs
        :param show_pbar: whether or not to show a progress bar
        """
        self.func = func
        self.df_kwargs = df_kwargs
        self.devices = set(devices)
        self.show_pbar = show_pbar

        self.manager = Manager()
        self.kwargs = [{k: r[k] for k in df_kwargs.columns} for r in df_kwargs.to_records()]

    def get_gpu(self):
        self.semaphore.acquire()
        if len(self.live_processes) > 0:
            for name, proc in list(self.live_processes.items()):
                # results are populated before the process terminates
                # so this detects if the function is completed
                if 'device' in self.results[name]:
                    device = self.results[name]['device']
                    self.devices.add(device)
                    del self.live_processes[name]
                    self.done_processes[name] = proc
                    self.pbar.update()
            for name, proc in list(self.done_processes.items()):  # garbage collection
                if not proc.is_alive():
                    proc.close()
                    del self.done_processes[name]
        return self.devices.pop()

    def run(self):
        self.semaphore = Semaphore(len(self.devices))
        self.live_processes = {}
        self.done_processes = {}
        self.results = {}

        if self.show_pbar:
            self.pbar = tqdm(total=len(self.kwargs))

        for i, kwargs in zip(range(len(self.kwargs)), self.kwargs):
            device = self.get_gpu()
            return_dict = self.manager.dict()
            self.results[i] = return_dict
            proc = Process(target=run_func_gpu,
                           args=(self.func, device, self.semaphore, return_dict),
                           kwargs=kwargs,
                           name=i)
            self.live_processes[i] = proc
            proc.start()

        for p in self.live_processes.values():  # wait for remaining processes to finish
            p.join()
            p.close()
            self.pbar.update()
        return [self.results[i]['result'] for i in range(len(self.kwargs))]
